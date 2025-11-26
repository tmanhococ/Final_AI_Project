"""
Module Posture Analyzer - Phân tích tư thế sitting

Module này cung cấp các chức năng:
- Phân tích tư thế ngồi với MediaPipe Pose
- Tính toán các góc đầu và vai
- Ước tính khoảng cách đến camera
- Detect bad posture cho health monitoring
"""

from __future__ import annotations
import time
import cv2
import numpy as np
from collections import deque
from typing import Dict, Any, Optional, Tuple

import mediapipe as mp
from utils import get_config, get_camera_calibration


class PostureAnalyzer:
    """
    Lớp Posture Analyzer - Phân tích tư thế ngồi

    Sử dụng MediaPipe Pose để:
    - Detect 33 pose landmarks
    - Tính toán góc đầu và vai
    - Ước tính khoảng cách đến màn hình
    - Phân loại chất lượng tư thế

    Attributes:
        _pose: MediaPipe Pose object
        _focal: Camera focal length
        _avg_eye_cm: Khoảng cách trung bình giữa 2 mắt (cm)
        _yaw_filter/_pitch_filter/_shoulder_filter: Moving average filters
        _latest: Dict - Dữ liệu analysis mới nhất
    """

    def __init__(self, config_path: str = "settings.json"):
        """
        Khởi tạo Posture Analyzer

        Args:
            config_path: Đường dẫn đến file cấu hình
        """
        self.cfg = get_config(config_path)
        health_cfg = self.cfg.get("health_monitoring", {})

        if not health_cfg:
            raise ValueError("Invalid config: 'health_monitoring' section not found")

        # Khởi tạo MediaPipe Pose
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            min_detection_confidence=float(health_cfg["pose_detection_confidence"]),
            min_tracking_confidence=float(health_cfg["pose_tracking_confidence"]),
        )

        # Cấu hình camera và đo lường
        self._focal = float(health_cfg["camera_focal_length"] or get_camera_calibration()["focal_length"])
        self._avg_eye_cm = float(health_cfg["AVERAGE_EYE_DISTANCE_CM"])  # ~6.3cm trung bình
        self._min_eye_px = int(health_cfg["MIN_EYE_PIXEL_DISTANCE"])
        self._min_dist_cm = float(health_cfg["MIN_REASONABLE_DISTANCE"])
        self._max_dist_cm = float(health_cfg["MAX_REASONABLE_DISTANCE"])
        self._eps = float(health_cfg["EPSILON"])

        # Ngưỡng tư thế (độ)
        self._max_head_yaw = float(health_cfg["max_head_side_angle"])    # Góc quay ngang đầu
        self._max_head_pitch = float(health_cfg["max_head_updown_angle"])  # Góc nghiêng đầu lên/xuống
        self._max_shoulder_tilt = float(health_cfg["max_shoulder_tilt"])   # Góc nghiêng vai

        # Moving average filters để giảm nhiễu
        self._yaw_filter = deque(maxlen=5)
        self._pitch_filter = deque(maxlen=5)
        self._shoulder_filter = deque(maxlen=5)
        self._dist_filter = deque(maxlen=3)

        # Runtime state
        self._latest: Dict[str, Any] = {}

    def analyze(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Phân tích tư thế từ frame ảnh

        - Detect pose landmarks
        - Tính toán các góc và khoảng cách
        - Áp dụng moving average filter
        - Phân loại chất lượng tư thế

        Args:
            frame: Input image frame

        Returns:
            Dict: Kết quả phân tích bao gồm:
                - head_side_angle: Góc quay ngang đầu (độ)
                - head_updown_angle: Góc nghiêng đầu lên/xuống (độ)
                - shoulder_tilt: Góc nghiêng vai (độ)
                - eye_distance: Khoảng cách ước tính đến màn hình (cm)
                - status: "good", "poor", hoặc "unknown"
        """
        if frame is None:
            return self._empty_result()

        h, w = frame.shape[:2]

        # Convert RGB cho MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._pose.process(rgb)
        rgb.flags.writeable = True

        # Không detect được pose
        if not results.pose_landmarks:
            return self._empty_result()

        lm = results.pose_landmarks.landmark

        # Trích xuất key landmarks
        left_eye = self._landmark_xyz(lm, mp.solutions.pose.PoseLandmark.LEFT_EYE, w, h)
        right_eye = self._landmark_xyz(lm, mp.solutions.pose.PoseLandmark.RIGHT_EYE, w, h)
        left_shoulder = self._landmark_xyz(lm, mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, w, h)
        right_shoulder = self._landmark_xyz(lm, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, w, h)
        nose = self._landmark_xyz(lm, mp.solutions.pose.PoseLandmark.NOSE, w, h)

        # Tính toán các vector
        eye_vec = right_eye[:2] - left_eye[:2]      # Vector từ mắt trái đến mắt phải
        shoulder_vec = right_shoulder[:2] - left_shoulder[:2]  # Vector từ vai trái đến vai phải
        head_vec = nose[:2] - (left_eye[:2] + right_eye[:2]) / 2  # Vector từ trung bình mắt đến mũi

        # Tính góc
        yaw = self._angle(eye_vec, (1, 0))           # Góc quay ngang đầu
        pitch = self._angle(head_vec, (0, -1))       # Góc nghiêng đầu lên/xuống
        shoulder_tilt = self._angle(shoulder_vec, (1, 0))  # Góc nghiêng vai

        # Ước tính khoảng cách tới màn hình
        eye_px = np.linalg.norm(right_eye[:2] - left_eye[:2])  # Khoảng cách mắt theo pixel
        distance_cm = None
        if eye_px >= self._min_eye_px:
            # Công thức: distance = (real_distance * focal_length) / pixel_distance
            distance_cm = (self._avg_eye_cm * self._focal) / eye_px
            distance_cm = np.clip(distance_cm, self._min_dist_cm, self._max_dist_cm)

        # Áp dụng moving average filter
        self._yaw_filter.append(yaw)
        self._pitch_filter.append(pitch)
        self._shoulder_filter.append(shoulder_tilt)
        if distance_cm is not None:
            self._dist_filter.append(distance_cm)

        # Lấy giá trị trung bình
        yaw_f = np.mean(self._yaw_filter)
        pitch_f = np.mean(self._pitch_filter)
        shoulder_f = np.mean(self._shoulder_filter)
        dist_f = np.mean(self._dist_filter) if self._dist_filter else None

        # Chuẩn hóa góc về quanh 0 độ [-90, +90]
        yaw_normalized = self._normalize_angle_to_zero(yaw_f)
        pitch_normalized = self._normalize_angle_to_zero(pitch_f)
        shoulder_normalized = self._normalize_angle_to_zero(shoulder_f)

        # Phân loại chất lượng tư thế (dùng góc đã chuẩn hóa)
        status = self._classify_normalized(yaw_normalized, pitch_normalized, shoulder_normalized, dist_f)

        # Cập nhật latest result với góc đã chuẩn hóa
        self._latest = {
            "timestamp": time.time(),
            "head_side_angle": yaw_normalized,
            "head_updown_angle": pitch_normalized,
            "shoulder_tilt": shoulder_normalized,
            "eye_distance_cm": dist_f,
            "status": status,
        }

        return self._latest

    def get_latest(self) -> Dict[str, Any]:
        """
        Lấy kết quả phân tích mới nhất

        Returns:
            Dict: Kết quả phân tích posture gần nhất
        """
        if not self._latest or "timestamp" not in self._latest:
            # Return empty result if no analysis has been run
            return self._empty_result()
        return self._latest.copy()

    @staticmethod
    def _landmark_xyz(lm, enum, w, h) -> np.ndarray:
        """
        Chuyển đổi landmark từ MediaPipe sang tọa độ pixel

        Args:
            lm: MediaPipe landmarks list
            enum: PoseLandmark enum value
            w: Image width
            h: Image height

        Returns:
            np.ndarray: Tọa độ [x, y, z] trong pixel
        """
        p = lm[enum.value]
        return np.array([p.x * w, p.y * h, p.z * w])

    @staticmethod
    def _angle(v1: np.ndarray, v2: Tuple[float, float]) -> float:
        """
        Tính góc giữa hai vector

        Args:
            v1: Vector thứ nhất
            v2: Vector thứ hai

        Returns:
            float: Góc tính bằng độ [0, 180]
        """
        denom = max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-7)
        cos_ang = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_ang)))

    @staticmethod
    def _normalize_angle_to_zero(angle: float) -> float:
        """
        Chuẩn hóa góc về quanh 0 độ trong khoảng [-90, +90]

        Ví dụ:
        - 170° → -10° (vì 170° gần -10° hơn là 170°)
        - 150° → -30°
        - 95° → -85°
        - 90° → 90° (giữ nguyên)
        - 45° → 45° (giữ nguyên)
        - 0° → 0° (giữ nguyên)

        Args:
            angle: Góc đầu vào [0, 180]

        Returns:
            float: Góc đã chuẩn hóa về [-90, +90]
        """
        # Giữ nguyên góc trong [-90, +90]
        if -90 <= angle <= 90:
            return angle

        # Với góc > 90, chuyển về [-90, 0] bằng cách trừ 180
        if angle > 90:
            return angle - 180

        # Với góc < -90, chuyển về [0, 90] bằng cách cộng 180
        return angle + 180

    def _classify_normalized(self, yaw: float, pitch: float, shoulder: float, dist: Optional[float]) -> str:
        """
        Phân loại chất lượng tư thế sử dụng góc đã chuẩn hóa

        Args:
            yaw: Góc quay ngang đầu đã chuẩn hóa [-90, +90]
            pitch: Góc nghiêng đầu đã chuẩn hóa [-90, +90]
            shoulder: Góc nghiêng vai đã chuẩn hóa [-90, +90]
            dist: Khoảng cách đến màn hình

        Returns:
            str: "good", "poor", hoặc "unknown"
        """
        # Handle None values trước khi kiểm tra
        if yaw is None or pitch is None or shoulder is None:
            return "unknown"

        # Với góc đã chuẩn hóa, chỉ cần kiểm tra giá trị tuyệt đối
        if abs(yaw) > self._max_head_yaw:
            return "poor"
        if abs(pitch) > self._max_head_pitch:
            return "poor"
        if abs(shoulder) > self._max_shoulder_tilt:
            return "poor"
        if dist is not None and (dist < self._min_dist_cm or dist > self._max_dist_cm):
            return "poor"

        return "good"

    def _classify(self, yaw: float, pitch: float, shoulder: float, dist: Optional[float]) -> str:
        """
        Phân loại chất lượng tư thế

        Kiểm tra các ngưỡng:
        - Góc quay đầu không quá max_head_yaw
        - Góc nghiêng đầu không quá max_head_pitch
        - Góc nghiêng vai không quá max_shoulder_tilt
        - Khoảng cách trong khoảng hợp lý

        Args:
            yaw: Góc quay ngang đầu
            pitch: Góc nghiêng đầu
            shoulder: Góc nghiêng vai
            dist: Khoảng cách đến màn hình

        Returns:
            str: "good", "poor", hoặc "unknown"
        """
        # Sử dụng hàm chuẩn hóa mới để一致 với analyze()
        # Handle None values trước khi chuẩn hóa
        if yaw is None or pitch is None or shoulder is None:
            return "unknown"

        yaw_norm = self._normalize_angle_to_zero(yaw)
        pitch_norm = self._normalize_angle_to_zero(pitch)
        shoulder_norm = self._normalize_angle_to_zero(shoulder)

        # Kiểm tra từng ngưỡng với góc đã chuẩn hóa
        if abs(yaw_norm) > self._max_head_yaw:
            return "poor"
        if abs(pitch_norm) > self._max_head_pitch:
            return "poor"
        if abs(shoulder_norm) > self._max_shoulder_tilt:
            return "poor"
        if dist is not None and (dist < self._min_dist_cm or dist > self._max_dist_cm):
            return "poor"

        return "good"

    def _empty_result(self) -> Dict[str, Any]:
        """
        Tạo kết quả rỗng khi không detect được pose

        Returns:
            Dict: Kết quả empty với các giá trị None
        """
        return {
            "timestamp": time.time(),
            "head_side_angle": None,
            "head_updown_angle": None,
            "shoulder_tilt": None,
            "eye_distance_cm": None,
            "status": "unknown",
        }

    def close(self) -> None:
        """
        Đóng MediaPipe Pose resources
        """
        self._pose.close()