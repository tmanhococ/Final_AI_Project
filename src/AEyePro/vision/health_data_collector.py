"""
Module Health Data Collector - Thu thập và lưu trữ dữ liệu sức khỏe

Module này cung cấp các chức năng:
- Thu thập dữ liệu realtime từ các vision modules
- Lưu trữ dữ liệu theo cấu trúc có tổ chức
- Tạo báo cáo summary theo session
- Quản lý vòng đời dữ liệu
"""

from __future__ import annotations
import time
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from utils import ExecutorService, append_csv_row, DATA_DIR

logger = logging.getLogger(__name__)


class HealthDataCollector:
    """
    Lớp Health Data Collector - Thu thập và quản lý dữ liệu sức khỏe

    Tổ chức lưu trữ:
    - realtime_YYYYMMDD_<session_id>.csv : lưu realtime (1 row / giây)
    - summary.csv : lưu summary (1 row / session)

    Attributes:
        collect_interval: Thời gian thu thập data (giây)
        session_id: ID unique cho session hiện tại
        executor: Thread pool cho async operations
    """

    def __init__(
        self,
        collect_interval: float = 1.0,
        config_path: str = "settings.json",
        executor: Optional[ExecutorService] = None,
    ):
        """
        Khởi tạo Health Data Collector

        Args:
            collect_interval: Khoảng thời gian thu thập data (giây)
            config_path: Đường dẫn file cấu hình
            executor: Thread pool executor (optional)
        """
        self.collect_interval = collect_interval
        self.session_id = str(uuid.uuid4())[:8]
        self.executor = executor or ExecutorService(max_workers=2)

        self.data_dir = DATA_DIR
        self.rt_csv_path = None
        self.summary_csv_path = self.data_dir / "summary.csv"

        self._running = False
        self._future = None
        self._start_ts = 0.0
        self._latest: Dict[str, Any] = {}
        self._reset_stats()

        # Đảm bảo data directory tồn tại
        self.data_dir.mkdir(exist_ok=True)

    # ----------------------------- public ----------------------------- #
    def start_collection(self) -> None:
        """
        Bắt đầu thu thập dữ liệu

        - Khởi tạo session ID
        - Tạo file CSV cho realtime data
        - Bắt đầu collection loop
        """
        if self._running:
            logger.warning("Collection already running")
            return

        self._running = True
        self._start_ts = time.time()
        self._reset_stats()

        # Tạo đường dẫn file realtime theo date
        date_str = datetime.now().strftime("%Y%m%d")
        self.rt_csv_path = self.data_dir / f"realtime_{date_str}_{self.session_id}.csv"

        # Bắt đầu collection loop
        self._future = self.executor.submit(self._loop)
        logger.info("HealthDataCollector started (session: %s)", self.session_id)

    def stop_collection(self) -> None:
        """
        Dừng thu thập dữ liệu

        - Dừng collection loop
        - Ghi summary statistics
        - Cleanup resources
        """
        if not self._running:
            return

        self._running = False

        if self._future:
            try:
                self._future.result(timeout=2.0)
            except Exception as e:
                logger.error("Error stopping collection: %s", e)

        # Ghi summary khi kết thúc
        self._write_summary()
        logger.info("HealthDataCollector stopped (session: %s)", self.session_id)

    def update_health_data(self, health_data: Dict[str, Any]) -> None:
        """
        Cập nhật health data mới

        - Đếm transitions (blink/drowsiness)
        - Lưu latest data
        - Cập nhật thống kê runtime

        Args:
            health_data: Dictionary chứa health metrics
        """
        # Đếm số lần chớp mắt và buồn ngủ dựa trên trạng thái chuyển từ False -> True
        blink_now = bool(health_data.get("blink_detected", False))
        drowsy_now = bool(health_data.get("drowsiness_detected", False))

        if blink_now and not self._last_blink_state:
            self.blink_count += 1

        if drowsy_now and not self._last_drowsy_state:
            self.drowsiness_count += 1

        self._last_blink_state = blink_now
        self._last_drowsy_state = drowsy_now
        self._latest = health_data.copy()

        # Cập nhật thống kê runtime
        self._update_runtime_stats(health_data)

    def get_current_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê hiện tại của session

        Returns:
            Dict: Thống kê realtime
        """
        session_duration = time.time() - self._start_ts if self._start_ts > 0 else 0

        return {
            "session_id": self.session_id,
            "session_duration_seconds": session_duration,
            "total_records": self.total_records,
            "total_blinks": self.blink_count,
            "total_drowsiness_events": self.drowsiness_count,
            "avg_distance_cm": self.sum_dist / self.total_records if self.total_records > 0 else 0,
            # Focus: 3 key posture angles
            "avg_shoulder_tilt_deg": self.sum_shoulder_tilt / self.total_records if self.total_records > 0 else 0,  # Góc vai
            "avg_head_pitch_deg": self.sum_head_pitch / self.total_records if self.total_records > 0 else 0,     # Góc đầu trước-sau
            "avg_head_yaw_deg": self.sum_head_yaw / self.total_records if self.total_records > 0 else 0,         # Góc đầu trái-phải
            "blink_rate_per_minute": (self.blink_count / session_duration * 60) if session_duration > 0 else 0,
        }

    # ----------------------------- internal --------------------------- #
    def _reset_stats(self):
        """
        Reset tất cả thống kê - Focus on 3 key posture angles
        """
        self.total_records = 0
        self.total_blinks = 0
        self.blink_count = 0
        self.drowsiness_count = 0
        self._last_blink_state = False
        self._last_drowsy_state = False

        # Focus: 3 key posture angles statistics
        self.sum_dist = 0.0
        self.sum_shoulder_tilt = 0.0    # Góc vai
        self.sum_head_pitch = 0.0       # Góc đầu trước-sau
        self.sum_head_yaw = 0.0         # Góc đầu trái-phải

    def _update_runtime_stats(self, health_data: Dict[str, Any]) -> None:
        """
        Cập nhật thống kê realtime từ health data - Focus on 3 key posture angles

        Args:
            health_data: Dictionary containing health metrics
        """
        # Cập nhật khoảng cách trung bình
        distance = health_data.get("distance_cm")
        if distance is not None:
            self.sum_dist += distance

        # Focus: Cập nhật 3 góc tư thể chính
        shoulder_tilt = health_data.get("shoulder_tilt", 0)
        head_pitch = health_data.get("head_pitch", 0)
        head_yaw = health_data.get("head_yaw", 0)

        self.sum_shoulder_tilt += abs(shoulder_tilt)    # Góc vai
        self.sum_head_pitch += abs(head_pitch)          # Góc đầu trước-sau
        self.sum_head_yaw += abs(head_yaw)              # Góc đầu trái-phải

        # Cập nhật các metrics khác
        self.total_records += 1

    def _loop(self) -> None:
        """
        Main collection loop - chạy trong thread riêng

        - Thu thập data theo collect_interval
        - Ghi vào realtime CSV file
        - Xử lý exceptions để đảm bảo stability
        """
        logger.info("Starting health data collection loop")

        while self._running:
            try:
                start_time = time.time()

                # Chuẩn bị data row cho CSV
                data_row = self._prepare_csv_row()

                # Ghi vào CSV file
                if self.rt_csv_path and data_row:
                    append_csv_row(data_row, str(self.rt_csv_path))
                    self.total_records += 1

                # Sleep để maintain collect_interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.collect_interval - elapsed)

                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    logger.warning("Collection took longer than interval (%.3fs)", elapsed)

            except Exception as e:
                logger.error("Error in collection loop: %s", e)
                time.sleep(self.collect_interval)  # Prevent tight error loop

        logger.info("Health data collection loop ended")

    def _prepare_csv_row(self) -> Dict[str, Any]:
        """
        Chuẩn bị data row cho CSV storage - OPTIMIZED for 3 key posture angles

        Returns:
            Dict: Row data với health metrics focused (20 → 10 fields)
        """
        timestamp = time.time()
        datetime_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

        # Lấy latest health data
        data = self._latest.copy()

        # Prepare optimized row with focus on health metrics
        row = {
            "timestamp": timestamp,
            "datetime": datetime_str,
            "session_id": self.session_id,
            "frame_timestamp": data.get("timestamp", timestamp),
            "processing_time_ms": data.get("proc_ms", 0),

            # Eye Health (optimized - only essential metrics)
            "avg_ear": data.get("avg_ear"),

            # Ergonomics
            "distance_cm": data.get("distance_cm"),

            # Focus: 3 key posture angles
            "shoulder_tilt": data.get("shoulder_tilt"),        # Góc vai
            "head_pitch": data.get("head_pitch"),             # Góc đầu trước-sau
            "head_yaw": data.get("head_yaw"),                 # Góc đầu trái-phải

            # Detection results (essential only)
            "blink_detected": data.get("blink_detected", False),
            "drowsiness_detected": data.get("drowsiness_detected", False),
            "posture_good": data.get("posture_good", True),
        }

        # Remove None values to keep CSV clean
        row = {k: v for k, v in row.items() if v is not None}

        return row

    def _write_summary(self) -> None:
        """
        Ghi summary statistics khi kết thúc session

        - Tính toán các metrics thống kê
        - Ghi vào summary.csv file
        """
        try:
            session_duration = time.time() - self._start_ts

            # Prepare summary row
            summary_row = {
                "session_id": self.session_id,
                "start_time": self._start_ts,
                "end_time": time.time(),
                "duration_seconds": session_duration,
                "start_datetime": datetime.fromtimestamp(self._start_ts).strftime("%Y-%m-%d %H:%M:%S"),
                "end_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

                # Collection statistics
                "total_records": self.total_records,
                "avg_records_per_minute": (self.total_records / session_duration * 60) if session_duration > 0 else 0,

                # Blink statistics
                "total_blinks": self.blink_count,
                "blink_rate_per_minute": (self.blink_count / session_duration * 60) if session_duration > 0 else 0,

                # Drowsiness statistics
                "drowsiness_events": self.drowsiness_count,

                # Average metrics - Focus on 3 key posture angles
                "avg_distance_cm": self.sum_dist / self.total_records if self.total_records > 0 else 0,
                "avg_shoulder_tilt_deg": self.sum_shoulder_tilt / self.total_records if self.total_records > 0 else 0,  # Góc vai
                "avg_head_pitch_deg": self.sum_head_pitch / self.total_records if self.total_records > 0 else 0,     # Góc đầu trước-sau
                "avg_head_yaw_deg": self.sum_head_yaw / self.total_records if self.total_records > 0 else 0,         # Góc đầu trái-phải

                # File information
                "realtime_file": str(self.rt_csv_path) if self.rt_csv_path else None,
            }

            # Ghi vào summary CSV
            append_csv_row(summary_row, str(self.summary_csv_path))
            logger.info("Summary written for session %s: %d records, %d blinks",
                       self.session_id, self.total_records, self.blink_count)

        except Exception as e:
            logger.error("Error writing summary: %s", e)

    def cleanup_old_data(self, retention_days: int = 7) -> None:
        """
        Xóa old data files theo retention policy

        Args:
            retention_days: Số ngày giữ lại data
        """
        try:
            cutoff_time = time.time() - (retention_days * 24 * 3600)
            deleted_files = []

            for file_path in self.data_dir.glob("realtime_*.csv"):
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    deleted_files.append(str(file_path))

            if deleted_files:
                logger.info("Deleted %d old data files", len(deleted_files))
            else:
                logger.info("No old data files to delete")

        except Exception as e:
            logger.error("Error cleaning up old data: %s", e)