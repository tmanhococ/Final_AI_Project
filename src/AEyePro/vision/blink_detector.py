"""
Module Blink Detector - Phát hiện và phân tích nháy mắt

Module này cung cấp các chức năng:
- Phát hiện nháy mắt dựa trên Eye Aspect Ratio (EAR)
- Lọc nhiễu khi quay đầu using moving window
- Đếm số lần nháy mắt và tính blink rate
- Phân loại các loại nháy mắt (normal, slow, forced)
- Cung cấp thống kê chi tiết cho health monitoring
"""

from __future__ import annotations
import time
from typing import Tuple, Optional, Any
from collections import deque

from utils import get_config
from vision.eye_tracker import EyeTracker


class BlinkDetector:
    """
    Lớp Blink Detector - Phát hiện và phân tích nháy mắt

    Sử dụng các thuật toán:
    - EAR-based detection với ngưỡng động
    - Moving window filter để loại nhiễu
    - Temporal analysis để detect blink patterns
    - Head movement compensation

    Attributes:
        cfg: Dict - Cấu hình từ file settings
        eye_tracker: EyeTracker - Source cho eye data
        blink_count: int - Tổng số lần nháy đã đếm
    """

    def __init__(
        self,
        config_path: str = "settings.json",
        eye_tracker: Optional[EyeTracker] = None,
    ):
        """
        Khởi tạo Blink Detector

        Args:
            config_path: Đường dẫn file cấu hình
            eye_tracker: EyeTracker instance để lấy data
        """
        self.cfg = get_config(config_path)
        self.eye_tracker = eye_tracker
        health_cfg = self.cfg.get("health_monitoring", {})

        # --- Tham số từ settings.json ---
        self.consecutive_frames = int(health_cfg["consecutive_frames"])  # số frame nhắm liên tiếp để tính là blink
        self.ear_th = float(health_cfg["BLINK_THRESHOLD"])
        self.max_blink_dur = float(health_cfg["max_blink_duration"])
        self.min_blink_gap = float(health_cfg["min_blink_interval"])
        self.max_head_yaw = float(health_cfg["max_head_side_angle"])
        self.max_head_pitch = float(health_cfg["max_head_updown_angle"])

        # --- Trạng thái runtime ---
        self.blink_count = 0
        self._closed_frames = 0
        self._blink_start = 0.0
        self._last_blink_ts = 0.0
        self._blink_end_time = 0.0

        # --- Statistics ---
        self.blink_durations = deque(maxlen=100)  # Lưu 100 blink durations gần nhất
        self.blink_intervals = deque(maxlen=100)  # Lưu 100 inter-blink intervals
        self.session_start_time = time.time()

        # --- Queue chống nhiễu khi quay đầu ---
        self._yaw_queue: deque[float] = deque(maxlen=30)  # ~1 giây @ 30fps
        self._yaw_window = 1.0  # giây
        self._counting_active = False

        # --- State tracking ---
        self._current_ear = None
        self._last_ear = None
        self._ear_buffer = deque(maxlen=5)  # De-bounce buffer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(self) -> dict[str, Any]:
        """
        Cập nhật và phát hiện nháy mắt từ eye tracker data

        Gọi từ Manager hoặc vòng lặp chính.
        Trả về dict kết quả gọn gàng cho GUI / ChartManager / Database.

        Returns:
            Dict: Kết quả blink detection với đầy đủ thông tin
        """
        data = self.eye_tracker.get_latest() if self.eye_tracker else {}
        frame = data.get("frame")
        avg_ear = data.get("avg_ear")
        head_pitch = data.get("head_pitch")
        head_yaw = data.get("head_yaw")
        avg_contrast = data.get("avg_contrast")

        blink, info = self._detect(avg_ear, avg_contrast, head_pitch, head_yaw)

        # Thêm thông tin bổ sung
        info.update({
            "frame": frame,
            "blink_detected": blink,
            "total_blinks": self.blink_count,
            "ear": avg_ear,
            "blink_rate_per_minute": self._calculate_blink_rate(),
            "avg_blink_duration": self._calculate_avg_blink_duration(),
            "avg_blink_interval": self._calculate_avg_blink_interval(),
        })

        return info

    def get_statistics(self) -> dict[str, Any]:
        """
        Lấy thống kê chi tiết về nháy mắt

        Returns:
            Dict: Thống kê blink detection
        """
        session_duration = time.time() - self.session_start_time
        return {
            "session_duration_minutes": session_duration / 60,
            "total_blinks": self.blink_count,
            "blink_rate_per_minute": self._calculate_blink_rate(),
            "avg_blink_duration_ms": self._calculate_avg_blink_duration() * 1000,
            "avg_blink_interval_ms": self._calculate_avg_blink_interval() * 1000,
            "last_blink_timestamp": self._last_blink_ts,
        }

    def reset_statistics(self) -> None:
        """
        Reset thống kê - dùng khi bắt đầu session mới
        """
        self.blink_count = 0
        self.session_start_time = time.time()
        self._last_blink_ts = 0.0
        self.blink_durations.clear()
        self.blink_intervals.clear()

    # ------------------------------------------------------------------
    # Blink Detection Logic
    # ------------------------------------------------------------------
    def _detect(
        self,
        ear: float | None,
        contrast: float | None,
        pitch: float | None,
        yaw: float | None,
    ) -> Tuple[bool, dict[str, Any]]:
        """
        Core blink detection algorithm

        Sử dụng kết hợp:
        - EAR threshold detection
        - Contrast analysis
        - Head movement compensation
        - Temporal filtering

        Args:
            ear: Eye Aspect Ratio
            contrast: Độ tương phản vùng mắt
            pitch: Góc nghiêng lên/xuống của đầu
            yaw: Góc nghiêng trái/phải của đầu

        Returns:
            Tuple[bool, Dict]: (blink_detected, detailed_info)
        """
        now = time.time()
        blinked = False
        reason = None

        # --- 1. Validate input ---
        if ear is None:
            self._reset_state()
            return False, {"reason": "no_ear_data"}

        # EAR de-bounce để loại nhiễu
        self._ear_buffer.append(ear)
        if len(self._ear_buffer) < 3:
            return False, {"reason": "insufficient_data"}

        # Sử dụng median để chống nhiễu
        filtered_ear = sorted(self._ear_buffer)[len(self._ear_buffer) // 2]
        self._current_ear = filtered_ear

        # --- 2. Head movement validation ---
        if pitch is not None and abs(pitch) > self.max_head_pitch:
            reason = "head_pitch_exceeded"
            self._reset_state()
            return False, {"reason": reason}

        if yaw is not None:
            # Moving average filter cho yaw
            self._yaw_queue.append(yaw)
            avg_yaw = sum(self._yaw_queue) / len(self._yaw_queue)

            if abs(avg_yaw) > self.max_head_yaw:
                reason = "head_yaw_exceeded"
                self._reset_state()
                return False, {"reason": reason}

        # --- 3. Blink detection ---
        eye_closed = filtered_ear < self.ear_th

        if eye_closed:
            # Mắt đang nhắm
            if not self._counting_active:
                # Bắt đầu một blink mới
                self._counting_active = True
                self._blink_start = now
                self._closed_frames = 1
            else:
                # Đang tiếp tục blink
                self._closed_frames += 1

                # Kiểm tra thời gian tối đa
                duration = now - self._blink_start
                if duration > self.max_blink_dur:
                    reason = "blink_too_long"
                    self._reset_state()
                    return False, {"reason": reason}
        else:
            # Mắt đang mở
            if self._counting_active:
                # Kết thúc một blink
                blink_duration = now - self._blink_start

                # Kiểm tra điều kiện để tính là valid blink
                if (self._closed_frames >= self.consecutive_frames and
                    blink_duration >= self.min_blink_gap):

                    # Kiểm tra khoảng cách với blink trước đó
                    time_since_last = now - self._last_blink_ts
                    if time_since_last >= self.min_blink_gap:
                        blinked = True
                        self.blink_count += 1
                        self._last_blink_ts = now
                        self._blink_end_time = now

                        # Lưu thống kê
                        self.blink_durations.append(blink_duration)
                        if self._last_blink_ts > 0:
                            self.blink_intervals.append(time_since_last)

                        reason = "valid_blink"
                    else:
                        reason = "too_soon_after_previous"

                # Reset state
                self._counting_active = False
                self._closed_frames = 0

        # --- 4. Build result ---
        info = {
            "reason": reason,
            "ear_filtered": filtered_ear,
            "ear_original": ear,
            "eye_closed": eye_closed,
            "closed_frames": self._closed_frames if self._counting_active else 0,
            "blink_duration": (now - self._blink_start) if self._counting_active else 0,
            "time_since_last_blink": now - self._last_blink_ts if self._last_blink_ts > 0 else None,
        }

        return blinked, info

    def _reset_state(self) -> None:
        """
        Reset internal state khi có invalid condition
        """
        self._counting_active = False
        self._closed_frames = 0
        self._ear_buffer.clear()
        self._yaw_queue.clear()

    def _calculate_blink_rate(self) -> float:
        """
        Tính blink rate (blinks/minute)

        Returns:
            float: Blink rate per minute
        """
        session_duration = time.time() - self.session_start_time
        if session_duration < 60:  # Chưa đủ 1 phút
            return 0.0
        return (self.blink_count / session_duration) * 60

    def _calculate_avg_blink_duration(self) -> float:
        """
        Tính average blink duration

        Returns:
            float: Average duration in seconds
        """
        if not self.blink_durations:
            return 0.0
        return sum(self.blink_durations) / len(self.blink_durations)

    def _calculate_avg_blink_interval(self) -> float:
        """
        Tính average interval giữa các blinks

        Returns:
            float: Average interval in seconds
        """
        if not self.blink_intervals:
            return 0.0
        return sum(self.blink_intervals) / len(self.blink_intervals)

    def analyze_blink_pattern(self) -> dict[str, Any]:
        """
        Phân tích pattern của nháy mắt để detect fatigue

        Returns:
            Dict: Phân tích blink pattern
        """
        if len(self.blink_durations) < 10:  # Cần đủ data samples
            return {
                "fatigue_indicators": {},
                "pattern": "insufficient_data"
            }

        # Calculate statistics
        avg_duration = self._calculate_avg_blink_duration()
        avg_interval = self._calculate_avg_blink_interval()
        blink_rate = self._calculate_blink_rate()

        # Fatigue indicators
        fatigue_indicators = {
            "slow_blinks": sum(1 for d in self.blink_durations if d > 0.4) / len(self.blink_durations),
            "frequent_blinks": blink_rate > 25,  # > 25 blinks/minute
            "irregular_rhythm": self._calculate_rhythm_irregularity(),
        }

        # Determine pattern
        if fatigue_indicators["slow_blinks"] > 0.3:
            pattern = "fatigue_detected"
        elif fatigue_indicators["frequent_blinks"]:
            pattern = "stress_detected"
        elif fatigue_indicators["irregular_rhythm"] > 0.5:
            pattern = "irregular_pattern"
        else:
            pattern = "normal"

        return {
            "fatigue_indicators": fatigue_indicators,
            "pattern": pattern,
            "blink_rate_per_minute": blink_rate,
            "avg_duration_ms": avg_duration * 1000,
            "avg_interval_ms": avg_interval * 1000,
        }

    def _calculate_rhythm_irregularity(self) -> float:
        """
        Tính độ không đều của rhythm nháy mắt

        Returns:
            float: Coefficient of variation (0-1)
        """
        if len(self.blink_intervals) < 5:
            return 0.0

        import statistics
        mean_interval = statistics.mean(self.blink_intervals)
        if mean_interval == 0:
            return 0.0

        std_interval = statistics.stdev(self.blink_intervals)
        return min(1.0, std_interval / mean_interval)