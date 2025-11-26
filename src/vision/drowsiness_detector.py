"""
Module Drowsiness Detector - Phát hiện buồn ngủ và mệt mỏi

Module này cung cấp các chức năng:
- Phát hiện buồn ngủ dựa trên Eye Aspect Ratio kéo dài
- Phân tích tư thế xấu kéo dài (ngủ gục, ngả người)
- Detect gaze off (nhìn ra xa) trong thời gian dài
- Kết hợp multiple signals với hysteresis để tránh false positives
"""

from __future__ import annotations
import time
import numpy as np
from collections import deque
from typing import Dict, Any, Optional

from utils import get_config


class DrowsinessDetector:
    """
    Lớp Drowsiness Detector - Phát hiện buồn ngủ

    Sử dụng multiple indicators:
    - EAR thấp kéo dài (mắt nhắm/mệt mỏi)
    - Tư thế xấu kéo dài (ngủ gục, nghiêng người)
    - Gaze off (nhìn ra xa hoặc không tập trung)
    - Hysteresis để duy trì trạng thái drowsy

    Attributes:
        ear_th: Ngưỡng EAR để detect mệt mỏi
        ear_duration_th: Thời gian tối thiểu EAR thấp để tính là buồn ngủ
        posture_window_sec: Thời gian tư thế xấu để tính là buồn ngủ
        gaze_off_threshold_sec: Thời gian gaze off để tính là buồn ngủ
        _drowsy: Trạng thái buồn ngủ hiện tại
    """

    def __init__(self, config_path: str = "settings.json"):
        """
        Khởi tạo Drowsiness Detector

        Args:
            config_path: Đường dẫn đến file cấu hình
        """
        self.cfg = get_config(config_path)
        health_cfg = self.cfg.get("health_monitoring", {})

        if not health_cfg:
            raise ValueError("Invalid config: 'health_monitoring' section not found")

        # Ngưỡng detect
        self.ear_th = float(health_cfg["DROWSY_THRESHOLD"])                       # Ngưỡng EAR
        self.ear_duration_th = float(health_cfg.get("drowsy_ear_duration", 2.0))  # Thời gian EAR thấp
        self.max_head_pitch = float(health_cfg["max_head_updown_angle"])           # Góc nghiêng đầu tối đa
        self.max_head_yaw = float(health_cfg["max_head_side_angle"])               # Góc quay đầu tối đa
        self.max_shoulder_tilt = float(health_cfg["max_shoulder_tilt"])            # Góc nghiêng vai tối đa
        self.min_gaze_distance_cm = float(health_cfg["MIN_REASONABLE_DISTANCE"])   # Khoảng cách tối thiểu
        self.max_gaze_distance_cm = float(health_cfg["MAX_REASONABLE_DISTANCE"])   # Khoảng cách tối đa

        # Thời gian threshold
        self.posture_window_sec = 3.0      # Tư thế xấu trong 3s
        self.gaze_off_threshold_sec = 2.0  # Gaze off trong 2s

        # EAR filtering
        self._ear_buf = deque(maxlen=5)            # Buffer cho moving average
        self._ear_low_start: Optional[float] = None  # Thời điểm bắt đầu EAR thấp
        self._ear_low_frames: int = 0               # Số frame liên tục EAR thấp
        self.EAR_CONSEC_FRAMES: int = 3             # ~100ms ở 30 FPS

        # Posture & Gaze tracking
        self._posture_bad_start: Optional[float] = None  # Thời điểm bắt đầu tư thế xấu
        self._gaze_off_start: Optional[float] = None     # Thời điểm bắt đầu gaze off
        self._gaze_last_seen: Optional[float] = None     # Lần cuối thấy gaze hợp lệ
        self.MAX_MISSING_DIST_SEC: float = 1.0           # Thời gian mất distance tối đa

        # Drowsiness state với hysteresis
        self._drowsy: bool = False
        self._drowsy_end_time: Optional[float] = None    # Thời điểm kết thúc drowsy
        self.DROWSY_RELEASE_SEC: float = 0.5             # Giữ trạng thái drowsy thêm 0.5s (nhạy hơn)

    def update(
        self,
        ear: Optional[float] = None,
        posture_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Cập nhật và detect buồn ngủ từ data mới

        Args:
            ear: Eye Aspect Ratio value
            posture_data: Dict từ PostureAnalyzer

        Returns:
            Dict: Kết quả drowsiness detection bao gồm:
                - drowsiness_detected: bool - có detect buồn ngủ không
                - reason: str - lý do detect (EAR + Posture, etc.)
                - ear_duration: float - thời gian EAR thấp
                - posture_bad_duration: float - thời gian tư thế xấu
                - gaze_off_duration: float - thời gian gaze off
        """
        now = time.time()

        # Initialize result
        info: Dict[str, Any] = {
            "drowsiness_detected": False,
            "reason": None,
            "ear_duration": 0.0,
            "posture_bad_duration": 0.0,
            "gaze_off_duration": 0.0,
        }

        # 1) Phân tích EAR với debounce
        ear_drowsy = self._analyze_ear(ear, now, info)

        # 2) Phân tích tư thế xấu
        posture_drowsy = self._analyze_posture(posture_data, now, info)

        # 3) Phân tích gaze off
        gaze_drowsy = self._analyze_gaze_off(posture_data, now, info)

        # 4) Tổng hợp các signals với hysteresis
        drowsy_signals = sum([ear_drowsy, posture_drowsy, gaze_drowsy])
        self._update_drowsy_state(drowsy_signals, now, info)

        info["drowsiness_detected"] = self._drowsy
        return info

    def _analyze_ear(self, ear: Optional[float], now: float, info: Dict[str, Any]) -> bool:
        """
        Phân tích EAR để detect mệt mỏi

        Args:
            ear: Eye Aspect Ratio value
            now: Timestamp hiện tại
            info: Dict để lưu kết quả

        Returns:
            bool: True nếu detect được drowsiness từ EAR
        """
        if ear is None:
            self._ear_low_start = None
            self._ear_low_frames = 0
            return False

        # Áp dụng moving average filter
        ear_f = self._filter_ear(ear, self._ear_buf)
        ear_low_now = ear_f < self.ear_th

        if ear_low_now:
            # EAR đang thấp
            self._ear_low_frames += 1

            # Bắt đầu đếm thời gian sau đủ frames liên tục
            if self._ear_low_start is None and self._ear_low_frames >= self.EAR_CONSEC_FRAMES:
                self._ear_low_start = now

            if self._ear_low_start is not None:
                info["ear_duration"] = now - self._ear_low_start
                return info["ear_duration"] >= self.ear_duration_th
        else:
            # EAR trở lại bình thường
            self._ear_low_start = None
            self._ear_low_frames = 0

        return False

    def _analyze_posture(self, posture_data: Optional[Dict[str, Any]], now: float, info: Dict[str, Any]) -> bool:
        """
        Phân tích tư thế để detect buồn ngủ

        Args:
            posture_data: Dict từ PostureAnalyzer
            now: Timestamp hiện tại
            info: Dict để lưu kết quả

        Returns:
            bool: True nếu detect được drowsiness từ tư thế
        """
        if posture_data is None:
            self._posture_bad_start = None
            return False

        # Extract posture metrics
        yaw = posture_data.get("head_side_angle")
        pitch = posture_data.get("head_updown_angle")
        shoulder = posture_data.get("shoulder_tilt")
        dist_cm = posture_data.get("eye_distance_cm")

        # Kiểm tra tư thế xấu
        posture_bad = False
        for val, limit in [
            (yaw, self.max_head_yaw),
            (pitch, self.max_head_pitch),
            (shoulder, self.max_shoulder_tilt),
        ]:
            if val is not None and abs(val) > limit:
                posture_bad = True

        # Kiểm tra khoảng cách不合理
        if dist_cm is not None and (
            dist_cm < self.min_gaze_distance_cm or dist_cm > self.max_gaze_distance_cm
        ):
            posture_bad = True

        if posture_bad:
            # Bắt đầu đếm thời gian tư thế xấu
            if self._posture_bad_start is None:
                self._posture_bad_start = now

            dur = now - self._posture_bad_start
            info["posture_bad_duration"] = dur
            return dur >= self.posture_window_sec
        else:
            # Tư thế trở lại bình thường
            self._posture_bad_start = None
            return False

    def _analyze_gaze_off(self, posture_data: Optional[Dict[str, Any]], now: float, info: Dict[str, Any]) -> bool:
        """
        Phân tích gaze off để detect mất tập trung

        Args:
            posture_data: Dict từ PostureAnalyzer
            now: Timestamp hiện tại
            info: Dict để lưu kết quả

        Returns:
            bool: True nếu detect được drowsiness từ gaze off
        """
        if posture_data is None:
            return False

        dist_cm = posture_data.get("eye_distance_cm")

        # Không có distance hoặc distance quá gần
        if dist_cm is None or dist_cm < self.min_gaze_distance_cm:
            # Nếu mất distance quá lâu → reset timer
            if dist_cm is None and self._gaze_off_start is not None:
                if now - self._gaze_off_start > self.MAX_MISSING_DIST_SEC:
                    self._gaze_off_start = None
            elif dist_cm is not None and dist_cm < self.min_gaze_distance_cm:
                # Distance quá gần (người ngả gần màn hình)
                if self._gaze_off_start is None:
                    self._gaze_off_start = now

                off_dur = now - self._gaze_off_start
                info["gaze_off_duration"] = off_dur
                return off_dur >= self.gaze_off_threshold_sec
        else:
            # Distance hợp lệ
            self._gaze_off_start = None

        return False

    def _update_drowsy_state(self, drowsy_signals: int, now: float, info: Dict[str, Any]) -> None:
        """
        Cập nhật trạng thái drowsiness với hysteresis - TĂNG NHẠY

        - Chỉ cần 1 signal để trigger drowsy (EAR thấp là đủ)
        - Giữ trạng thái drowsy thêm 0.5s sau khi signals mất
        - Tránh flickering trạng thái

        Args:
            drowsy_signals: Số signals hiện tại (0-3)
            now: Timestamp hiện tại
            info: Dict để lưu lý do
        """
        if drowsy_signals >= 1:
            # Trigger drowsiness
            self._drowsy = True
            self._drowsy_end_time = None
        else:
            # Bắt đầu hysteresis timer
            if self._drowsy and self._drowsy_end_time is None:
                self._drowsy_end_time = now

            # Kết thúc drowsiness sau hysteresis period
            if self._drowsy_end_time and now - self._drowsy_end_time >= self.DROWSY_RELEASE_SEC:
                self._drowsy = False
                self._drowsy_end_time = None

        # Ghi lý do nếu đang drowsy
        if self._drowsy and drowsy_signals >= 1:
            info["reason"] = self._get_drowsiness_reason(drowsy_signals)

    def _get_drowsiness_reason(self, drowsy_signals: int) -> str:
        """
        Lấy lý do drowsiness dựa trên số signals

        Args:
            drowsy_signals: Số signals (ear_drowsy + posture_drowsy + gaze_drowsy)

        Returns:
            str: Lý do drowsiness
        """
        # Đây là simplified version, trong thực tế cần track từng signal
        reasons = {
            3: "All Signals",
            2: "Combined Signals"
        }
        return reasons.get(drowsy_signals, "Unknown")

    def reset(self) -> None:
        """
        Reset tất cả trạng thái về giá trị ban đầu
        """
        self._ear_buf.clear()
        self._ear_low_start = None
        self._ear_low_frames = 0
        self._posture_bad_start = None
        self._gaze_off_start = None
        self._gaze_last_seen = None
        self._drowsy = False
        self._drowsy_end_time = None

    @staticmethod
    def _filter_ear(val: float, buf: deque) -> float:
        """
        Áp dụng weighted moving average filter cho EAR

        Args:
            val: Giá trị EAR mới
            buf: Buffer cũ

        Returns:
            float: Giá trị đã filter
        """
        buf.append(val)
        if len(buf) < 2:
            return val

        # Weighted average (weights tăng dần)
        weights = np.linspace(0.5, 1.0, len(buf))
        weights /= weights.sum()
        return float(np.average(buf, weights=weights))

    def reload_threshold(self, config_path: str = "settings.json") -> None:
        """
        Tải lại ngưỡng từ file config

        Args:
            config_path: Đường dẫn đến file config
        """
        cfg = get_config(config_path)
        health_cfg = cfg["health_monitoring"]
        self.ear_th = float(health_cfg["DROWSY_THRESHOLD"])

    def is_drowsy(self) -> bool:
        """
        Kiểm tra trạng thái drowsiness hiện tại

        Returns:
            bool: True nếu đang trong trạng thái drowsy
        """
        return self._drowsy