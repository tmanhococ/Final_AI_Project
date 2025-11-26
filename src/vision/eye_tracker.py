"""
Module Eye Tracker - Theo dõi và phân tích mắt sử dụng MediaPipe

Module này cung cấp các chức năng:
- Khởi tạo và quản lý camera với threading
- Phát hiện và tracking khuôn mặt, mắt bằng MediaPipe Face Mesh
- Tính toán Eye Aspect Ratio (EAR) để detect chớp mắt
- Phân tích độ tương phản của vùng mắt để đánh giá mắt mở
- Cung cấp data real-time cho các module khác với thread safety
"""

from __future__ import annotations
import cv2
import numpy as np
import mediapipe as mp
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from utils import get_config, ExecutorService, save_data


class EyeTracker:
    """
    Lớp Eye Tracker - Theo dõi và phân tích đặc điểm mắt

    Sử dụng MediaPipe Face Mesh để:
    - Phát hiện 468 landmarks của khuôn mặt
    - Trích xuất 6 landmarks cho mỗi mắt
    - Tính toán Eye Aspect Ratio (EAR)
    - Phân tích độ tương phản vùng mắt
    - Tính toán gaze point estimation

    Attributes:
        cfg: Dict - Cấu hình từ file settings
        _face_mesh: MediaPipe Face Mesh object
        _cap: OpenCV VideoCapture object
        _latest: Dict - Dữ liệu processing frame mới nhất
        _running: bool - Trạng thái hoạt động
    """

    # Class-level executor để chia sẻ giữa các instance
    _executor = ExecutorService(max_workers=2)

    def __init__(self, config_path: str | Path = "settings.json"):
        """
        Khởi tạo Eye Tracker với MediaPipe Face Mesh

        Args:
            config_path: Đường dẫn đến file cấu hình
        """
        self.cfg = get_config(str(config_path))
        health_cfg = self.cfg.get("health_monitoring", {})

        if not health_cfg:
            raise ValueError("Invalid config: 'health_monitoring' section not found")

        # Khởi tạo MediaPipe Face Mesh với high precision
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=float(health_cfg["min_detection_confidence"]),
            min_tracking_confidence=float(health_cfg["min_tracking_confidence"]),
        )

        # Camera configuration
        self._cap: Optional[cv2.VideoCapture] = None
        self._camera_idx = int(health_cfg.get("camera_index", 0))
        self._frame_rate = int(health_cfg.get("frame_rate", 30))

        # Threading và data
        self._lock = threading.Lock()
        self._latest: Dict[str, Any] = {}
        self._running = False

        # Eye landmarks indices theo MediaPipe Face Mesh
        self._LEFT_EYE = health_cfg["LEFT_EYE"]
        self._RIGHT_EYE = health_cfg["RIGHT_EYE"]
        self._EPS = float(health_cfg["EPSILON"])

        # Camera calibration parameters for distance estimation
        self._focal_length = float(health_cfg.get("camera_focal_length", 600))
        self._avg_eye_distance_cm = float(health_cfg.get("AVERAGE_EYE_DISTANCE_CM", 6.3))

        # Frame buffer cho debugging
        self.f = None

    def start(self) -> None:
        """
        Bắt đầu tracking mắt

        - Mở camera theo cấu hình
        - Khởi tạo thread processing
        - Bắt đầu capture loop

        Raises:
            RuntimeError: Nếu không thể mở camera
        """
        if self._running:
            return

        # Mở camera
        self._cap = cv2.VideoCapture(self._camera_idx)
        if not self._cap.isOpened():
            raise RuntimeError(f"Không thể mở camera {self._camera_idx}")

        # Cấu hình FPS
        self._cap.set(cv2.CAP_PROP_FPS, self._frame_rate)

        # Bắt đầu processing
        self._running = True
        EyeTracker._executor.submit(self._capture_loop)

    def stop(self) -> None:
        """
        Dừng tracking mắt

        - Dừng processing thread
        - Giải phóng camera
        - Đóng MediaPipe resources
        """
        if not self._running:
            return

        self._running = False
        self._cleanup()

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Lấy frame mới nhất từ camera

        Returns:
            np.ndarray: Frame hiện tại hoặc None nếu chưa có
        """
        return self.f

    def get_latest(self) -> Dict[str, Any]:
        """
        Lấy dữ liệu processing mới nhất với thread safety

        Returns:
            Dict: Copy của dữ liệu mới nhất
        """
        with self._lock:
            return self._latest.copy()

    def _capture_loop(self) -> None:
        """
        Main capture loop - chạy trong thread riêng

        - Đọc frame từ camera
        - Xử lý với MediaPipe
        - Lưu trữ kết quả với thread safety
        """
        while self._running and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret or frame is None:
                continue

            # Lưu frame gốc cho debugging
            self.f = frame.copy()

            # Xử lý frame
            data = self._process_frame(frame)

            # Thread-safe update
            with self._lock:
                self._latest = data

    def _process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Xử lý frame với MediaPipe Face Mesh

        - Chuyển đổi màu và xử lý với MediaPipe
        - Tính toán EAR cho cả hai mắt
        - Phân tích độ tương phản vùng mắt
        - Ước tính gaze point

        Args:
            frame: Input frame BGR

        Returns:
            Dict: Kết quả processing
        """
        start_time = time.perf_counter()

        # Chuyển đổi sang RGB cho MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._face_mesh.process(rgb)
        rgb.flags.writeable = True

        h, w, _ = frame.shape

        # Output structure
        out = {
            "frame": frame,
            "landmarks": None,
            "left_eye": None,
            "right_eye": None,
            "gaze_point": None,
            "left_ear": None,
            "right_ear": None,
            "avg_ear": None,
            "left_contrast": None,
            "right_contrast": None,
            "avg_contrast": None,
            "distance_cm": None,
            "timestamp": time.time(),
            "proc_ms": 0.0,
        }

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            # Trích xuất landmarks cho mắt
            l_pts = np.array([(lm[i].x * w, lm[i].y * h) for i in self._LEFT_EYE])
            r_pts = np.array([(lm[i].x * w, lm[i].y * h) for i in self._RIGHT_EYE])

            # Lưu landmarks
            out["landmarks"] = lm
            out["left_eye"] = l_pts
            out["right_eye"] = r_pts

            # Tính Eye Aspect Ratio (EAR)
            left_ear = self._calculate_ear(l_pts)
            right_ear = self._calculate_ear(r_pts)
            avg_ear = (left_ear + right_ear) / 2.0

            out["left_ear"] = left_ear
            out["right_ear"] = right_ear
            out["avg_ear"] = avg_ear

            # Phân tích độ tương phản vùng mắt
            left_contrast = self._calculate_eye_contrast(frame, l_pts)
            right_contrast = self._calculate_eye_contrast(frame, r_pts)
            avg_contrast = (left_contrast + right_contrast) / 2.0

            out["left_contrast"] = left_contrast
            out["right_contrast"] = right_contrast
            out["avg_contrast"] = avg_contrast

            # Ước tính khoảng cách đến camera
            eye_distance_px = np.linalg.norm(l_pts[0] - r_pts[0])
            if eye_distance_px > 30:  # Minimum reasonable pixel distance
                distance_cm = (self._avg_eye_distance_cm * self._focal_length) / eye_distance_px
                distance_cm = np.clip(distance_cm, 50, 120)  # Reasonable range
                out["distance_cm"] = distance_cm

            # Gaze point estimation (simplified - center of eyes)
            gaze_x = (l_pts[:, 0].mean() + r_pts[:, 0].mean()) / 2
            gaze_y = (l_pts[:, 1].mean() + r_pts[:, 1].mean()) / 2
            out["gaze_point"] = (gaze_x, gaze_y)

        # Processing time
        out["proc_ms"] = (time.perf_counter() - start_time) * 1000

        return out

    def _calculate_ear(self, eye_pts: np.ndarray) -> float:
        """
        Tính Eye Aspect Ratio (EAR)

        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

        Args:
            eye_pts: 6 eye landmarks (top-left, top, top-right, bottom-right, bottom, bottom-left)

        Returns:
            float: EAR value
        """
        if len(eye_pts) != 6:
            return 0.0

        # Vertical distances
        vert1 = np.linalg.norm(eye_pts[1] - eye_pts[5])  # p3-p5
        vert2 = np.linalg.norm(eye_pts[2] - eye_pts[4])  # p2-p6

        # Horizontal distance
        horiz = np.linalg.norm(eye_pts[0] - eye_pts[3])  # p1-p4

        # EAR calculation with epsilon to avoid division by zero
        ear = (vert1 + vert2) / (2.0 * horiz + self._EPS)

        return ear

    def _calculate_eye_contrast(self, frame: np.ndarray, eye_pts: np.ndarray) -> float:
        """
        Tính độ tương phản của vùng mắt để đánh giá mắt mở

        Sử dụng standard deviation của grayscale values trong ROI mắt

        Args:
            frame: Input frame BGR
            eye_pts: 6 eye landmarks

        Returns:
            float: Contrast value (0-255)
        """
        if len(eye_pts) != 6:
            return 0.0

        # Tạo ROI bao quanh mắt
        x_coords = eye_pts[:, 0]
        y_coords = eye_pts[:, 1]

        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())

        # Padding để đảm bảo lấy đủ vùng mắt
        padding = 5
        x_min = max(0, x_min - padding)
        x_max = min(frame.shape[1], x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(frame.shape[0], y_max + padding)

        # Trích xuất ROI và chuyển sang grayscale
        roi = frame[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            return 0.0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Tính standard deviation là measure của contrast
        contrast = float(np.std(gray))

        return contrast

    def _cleanup(self) -> None:
        """
        Dọn dẹp resources
        """
        if self._cap:
            self._cap.release()
            self._cap = None

        if self._face_mesh:
            self._face_mesh.close()

    def __del__(self):
        """
        Destructor - đảm bảo cleanup
        """
        self.stop()