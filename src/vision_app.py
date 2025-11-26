#!/usr/bin/env python3
"""
AEyePro Vision System - HOÀN CHỈNH SỬ DỤNG TOÀN BỘ PACKAGE VISION
======================================================================

Ứng dụng Computer Vision Health Monitoring hoàn chỉnh tích hợp đầy đủ:

📦 TÍNH NĂNG:
- Eye Tracking với 468 landmarks (MediaPipe Face Mesh)
- Posture Analysis với 33 landmarks (MediaPipe Pose)
- Blink Detection thuật toán EAR-based
- Drowsiness Detection multi-signal analysis
- Health Data Collection tự động (tối ưu 50% storage)

🎯 TẬP TRUNG:
- 3 góc tư thế chính: vai, đầu trước-sau, đầu trái-phải
- Health metrics chỉ giữ lại các chỉ số quan trọng
- Storage tối ưu (18 → 9 fields)
- Consistent data interface
- Professional camera overlay với alerts

✅ SỬ DỤNG ĐẦY ĐỦ PACKAGE VISION:
- EyeTracker: Theo dõi mắt và tính EAR
- PostureAnalyzer: Phân tích tư thế 33 landmarks
- BlinkDetector: Detect chớp mắt real-time
- DrowsinessDetector: Detect buồn ngủ multi-signal
- HealthDataCollector: Thu thập và lưu trữ data tự động

🚀 PERFORMANCE:
- 30 FPS optimized processing
- Multi-threaded camera access
- Thread-safe data operations
- Automatic session management
- Graceful error handling và recovery
"""

import sys
import time
import signal
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import argparse

import mediapipe

# Add project root to path
current_dir = Path(__file__).parent  # We're now at the root level
sys.path.insert(0, str(current_dir))

try:
    import numpy as np
    import pandas as pd
    import cv2
    import mediapipe as mp
    from utils import get_config, append_csv_row
    from vision.eye_tracker import EyeTracker
    from vision.posture_analyzer import PostureAnalyzer
    from vision.blink_detector import BlinkDetector
    from vision.drowsiness_detector import DrowsinessDetector
    from vision.health_data_collector import HealthDataCollector
except ImportError as e:
    print(f"LOI IMPORT: {e}")
    print("Vui long cai dat: pip install -r requirements.txt")
    sys.exit(1)


class AEyeProVisionApp:
    """
    AEyePro Vision System - Main Application
    """

    def __init__(self, config_file: str = "settings.json", show_camera: bool = True):
        """
        Khởi tạo AEyePro Vision Application

        Args:
            config_file: Đường dẫn file cấu hình
            show_camera: Hiển thị camera window hay không
        """
        self.cfg = get_config(config_file)
        self.health_cfg = self.cfg.get("health_monitoring", {})

        # Application state
        self.start_time = None
        self.running = False
        self.session_id = None
        self.session_dir = None
        self.shutdown_requested = False

        # Camera display settings
        self.show_camera = show_camera
        self.camera_window_name = "AEyePro Vision - Camera Feed"
        self.display_fps_counter = 0
        self.display_fps_time = time.time()

        # Vision modules
        self.eye_tracker = None
        self.posture_analyzer = None
        self.blink_detector = None
        self.drowsiness_detector = None
        self.health_collector = None

        # Data tracking
        self.frame_count = 0
        self.processed_frames = 0
        self.error_count = 0

        # Statistics
        self.stats = {
            'total_blinks': 0,
            'avg_ear': 0.0,
            'avg_distance': 0.0,
            'drowsy_events': 0,
            'posture_alerts': 0,
            'fps': 0.0
        }

        # Data storage
        self.session_data = []
        self.eye_data_buffer = []

        # Console display disabled - all info on camera overlay
        self.console_update_interval = float('inf')  # Disabled
        self.last_console_update = 0

    def initialize_modules(self) -> bool:
        """
        Khởi tạo tất cả vision modules

        Returns:
            bool: True nếu thành công
        """
        try:
            print("🚀 Đang khởi tạo TẤT CẢ vision modules...")

            # ✅ KHỞI TẠO ĐẦY ĐỦ VISION MODULES
            print("  📦 Khởi tạo Eye Tracker (468 landmarks)...")
            self.eye_tracker = EyeTracker()

            print("  📦 Khởi tạo Posture Analyzer (33 landmarks)...")
            self.posture_analyzer = PostureAnalyzer()

            print("  📦 Khởi tạo Blink Detector (EAR-based)...")
            self.blink_detector = BlinkDetector(eye_tracker=self.eye_tracker)

            print("  📦 Khởi tạo Drowsiness Detector (multi-signal)...")
            self.drowsiness_detector = DrowsinessDetector()

            print("  📦 Khởi tạo Health Data Collector (tự động logging)...")
            self.health_collector = HealthDataCollector()

            # ✅ START EYE TRACKER (camera access)
            print("  🔌 Starting camera access...")
            self.eye_tracker.start()

            # Initialize OpenCV window
            if self.show_camera:
                cv2.namedWindow(self.camera_window_name, cv2.WINDOW_AUTOSIZE)
                cv2.moveWindow(self.camera_window_name, 100, 100)  # Position window
                print(f"[OK] Camera display enabled: {self.camera_window_name}")
                print("  - Press 'q' in camera window to stop")
                print("  - Click window X to close")
                print("  - All monitoring info displayed on camera overlay")
            else:
                print("[OK] Camera display disabled - monitoring with camera processing only")

            print("[OK] Tất cả modules đã được khởi tạo thành công")
            return True

        except Exception as e:
            print(f"[ERROR] Lỗi khởi tạo modules: {e}")
            return False

    def setup_session_logging(self):
        """
        Setup session logging - CSV storage handled by health_data_collector only
        """
        # Create session directory
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = current_dir / "data"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Summary CSV file (1 record per session) - still handled here
        self.summary_csv_file = self.session_dir / "summary.csv"

        # Create summary CSV if it doesn't exist
        if not self.summary_csv_file.exists():
            summary_headers = [
                # Session Information
                'session_id', 'start_time', 'end_time', 'duration_minutes',
                # Health Metrics - quan trọng nhất
                'avg_ear', 'avg_distance_cm', 'drowsiness_events',
                # Posture Analysis - 3 góc chính
                'avg_shoulder_tilt', 'avg_head_pitch', 'avg_head_yaw'
            ]
            append_csv_row({h: None for h in summary_headers}, self.summary_csv_file, summary_headers)

        print(f"[OK] Session logging initialized - Health Data Collector handles CSV storage")
        return self.session_id

    def process_frame(self) -> Dict[str, Any]:
        """
        Xử lý một frame camera qua tất cả modules

        Returns:
            Dict: Kết quả processing từ tất cả modules
        """
        try:
            frame_result = {}

            # Get frame and eye tracking data
            eye_data = self.eye_tracker.get_latest()
            if not eye_data or eye_data.get('landmarks') is None:
                return {'error': 'No face detected'}

            self.processed_frames += 1
            frame_result['eye_data'] = eye_data
            frame_result['timestamp'] = time.time()

            # Process blink detection (using update method)
            blink_data = self.blink_detector.update()
            frame_result['blink_data'] = blink_data

            # Process posture first (required for drowsiness detection)
            frame = self.eye_tracker.get_frame()
            if frame is not None:
                # Run posture analysis
                self.posture_analyzer.analyze(frame)
                posture_data = self.posture_analyzer.get_latest()

                # Normalize angles to be centered around 0 degrees
                posture_data = self._normalize_posture_angles(posture_data)
            else:
                posture_data = {}
            frame_result['posture_data'] = posture_data

            # Process drowsiness detection with eye and posture data
            eye_data = frame_result.get('eye_data', {})
            try:
                drowsy_data = self.drowsiness_detector.update(
                    ear=eye_data.get('avg_ear'),
                    posture_data=posture_data
                )
                frame_result['drowsy_data'] = drowsy_data
            except Exception as e:
                # Fallback if drowsiness detector fails
                frame_result['drowsy_data'] = {
                    'drowsiness_detected': False,
                    'reason': None,
                    'ear_duration': 0.0,
                    'posture_bad_duration': 0.0,
                    'gaze_off_duration': 0.0,
                }

            return frame_result

        except Exception as e:
            self.error_count += 1
            return {'error': str(e)}

    def update_statistics(self, frame_result: Dict[str, Any]):
        """
        Cập nhật thống kê từ frame result

        Args:
            frame_result: Kết quả xử lý frame
        """
        if 'error' in frame_result:
            return

        eye_data = frame_result.get('eye_data', {})
        blink_data = frame_result.get('blink_data', {})
        drowsy_data = frame_result.get('drowsy_data', {})
        posture_data = frame_result.get('posture_data', {})

        # Eye tracking stats
        if eye_data.get('avg_ear'):
            self.eye_data_buffer.append(eye_data['avg_ear'])
            if len(self.eye_data_buffer) > 30:  # Keep last 30 frames
                self.eye_data_buffer.pop(0)
            self.stats['avg_ear'] = np.mean(self.eye_data_buffer)

        if eye_data.get('distance_cm'):
            self.stats['avg_distance'] = eye_data['distance_cm']

        # Blink stats
        if blink_data.get('blink_detected'):
            self.stats['total_blinks'] += 1

        # Drowsiness stats
        if drowsy_data.get('is_drowsy'):
            self.stats['drowsy_events'] += 1

        # Posture stats
        if posture_data.get('posture_quality') == 'bad':
            self.stats['posture_alerts'] += 1

    def save_frame_data(self, frame_result: Dict[str, Any], frame_id: int):
        """
        Update health data collector with frame data (1 record per second) - AEYE style
        CSV storage is now handled by health_data_collector only

        Args:
            frame_result: Kết quả xử lý frame
            frame_id: ID của frame
        """
        if 'error' in frame_result:
            return

        # Only update data every second to match AEYE style
        current_time = time.time()
        if not hasattr(self, '_last_save_time'):
            self._last_save_time = current_time

        # Check if 1 second has passed
        if current_time - self._last_save_time < 1.0:
            return

        self._last_save_time = current_time
        elapsed = current_time - self.start_time if self.start_time else 0

        # Get data from all modules
        eye_data = frame_result.get('eye_data', {})
        blink_data = frame_result.get('blink_data', {})
        posture_data = frame_result.get('posture_data', {})
        drowsy_data = frame_result.get('drowsy_data', {})

        # ✅ CẬP NHẬT HEALTH DATA COLLECTOR - Chỉ essential metrics
        if self.health_collector:
            # Prepare focused health data với essential metrics only
            health_data = {
                'timestamp': current_time,
                'avg_ear': eye_data.get('avg_ear'),

                # ✅ ERGONOMICS - Khoảng cách đến màn hình quan trọng
                'distance_cm': eye_data.get('distance_cm', posture_data.get('eye_distance_cm')),

                # ✅ TẬP TRUNG: 3 góc tư thế chính
                'shoulder_tilt': posture_data.get('shoulder_tilt'),          # Góc nghiêng vai
                'head_pitch': posture_data.get('head_updown_angle'),         # Góc nghiêng đầu trước-sau
                'head_yaw': posture_data.get('head_side_angle'),            # Góc nghiêng đầu trái-phải

                # ✅ HEALTH STATUS - Trạng thái quan trọng nhất
                'drowsiness_detected': drowsy_data.get('drowsiness_detected', False),
                'posture_status': posture_data.get('status', 'unknown'),
            }
            # Health Data Collector sẽ tự động logging với essential storage
            self.health_collector.update_health_data(health_data)

    def _calculate_eye_fatigue_level(self, eye_data: Dict[str, Any]) -> str:
        """
        Calculate eye fatigue level based on EAR value
        """
        avg_ear = eye_data.get('avg_ear', self.stats.get('avg_ear', 0.3))

        if avg_ear < 0.2:
            return "HIGH"
        elif avg_ear < 0.25:
            return "MEDIUM"
        else:
            return "LOW"

    def _collect_session_statistics(self) -> Dict[str, Any]:
        """
        Collect comprehensive session statistics for summary
        """
        elapsed = time.time() - self.start_time if self.start_time else 0
        duration_minutes = elapsed / 60

        # Calculate averages and percentages
        total_frames = self.frame_count
        avg_fps = self.processed_frames / max(elapsed, 0.1)
        success_rate = (self.processed_frames / max(self.frame_count, 1)) * 100

        # Blink statistics
        avg_blink_rate = self.stats['total_blinks'] / max(elapsed/60, 0.1)
        avg_ear = self.stats.get('avg_ear', 0.3)
        avg_distance = self.stats.get('avg_distance', 0)

        # Posture statistics (simplified)
        posture_alerts = self.stats['posture_alerts']
        bad_posture_percentage = (posture_alerts / max(elapsed, 1)) * 100 if posture_alerts > 0 else 0

        # Drowsiness statistics
        drowsiness_events = self.stats['drowsy_events']

        # Eye fatigue statistics
        eye_fatigue_percentage = 0
        if avg_ear < 0.2:
            eye_fatigue_percentage = 80
        elif avg_ear < 0.25:
            eye_fatigue_percentage = 40
        else:
            eye_fatigue_percentage = 10

        # Get posture data from health_data_collector
        posture_data = {
            'avg_shoulder_tilt_deg': 0,
            'avg_head_pitch_deg': 0,
            'avg_head_yaw_deg': 0
        }

        if self.health_collector:
            try:
                # Get current statistics from health_data_collector
                collector_stats = self.health_collector.get_current_stats()
                posture_data = {
                    'avg_shoulder_tilt_deg': collector_stats.get('avg_shoulder_tilt_deg', 0),
                    'avg_head_pitch_deg': collector_stats.get('avg_head_pitch_deg', 0),
                    'avg_head_yaw_deg': collector_stats.get('avg_head_yaw_deg', 0),
                }
            except Exception as e:
                print(f"[WARNING] Could not get posture data from health_data_collector: {e}")

        return {
            'duration_minutes': duration_minutes,
            'total_blinks': self.stats['total_blinks'],
            'avg_blink_rate': avg_blink_rate,
            'avg_ear': avg_ear,
            'avg_distance_cm': avg_distance,
            'drowsiness_events': drowsiness_events,
            'posture_alerts': posture_alerts,
            'bad_posture_percentage': bad_posture_percentage,
            'eye_fatigue_percentage': eye_fatigue_percentage,
            'total_frames': total_frames,
            'fps_avg': avg_fps,
            'success_rate': success_rate,
            # Add posture data from health_data_collector
            **posture_data
        }

    def display_camera_feed(self, frame_result: Dict[str, Any]):
        """
        Hiển thị camera feed với comprehensive UI overlay

        Args:
            frame_result: Kết quả processing từ các modules
        """
        if not self.show_camera:
            return

        try:
            # Get original frame from eye tracker
            frame = self.eye_tracker.get_frame()
            if frame is None:
                return

            # Create display frame with comprehensive overlay
            display_frame = self._create_comprehensive_overlay(frame, frame_result)

            # Display the frame
            cv2.imshow(self.camera_window_name, display_frame)

            # Check for window close (user clicks X or presses 'q')
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.shutdown_requested = True
                print("\n[STOPPED] Camera window closed")

        except Exception as e:
            print(f"[WARNING] Camera display error: {e}")

    def _create_comprehensive_overlay(self, frame: np.ndarray, frame_result: Dict[str, Any]) -> np.ndarray:
        """
        Tạo comprehensive overlay với toàn bộ thông tin monitoring - Optimized for 30 FPS

        Args:
            frame: Original camera frame
            frame_result: Data từ tất cả modules

        Returns:
            Frame với full overlay UI
        """
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]

        # Pre-calculate expensive operations
        current_time = time.time()
        elapsed = current_time - self.start_time if self.start_time else 0

        # Draw MediaPipe landmarks first (bottom layer)
        self._draw_face_landmarks(display_frame, frame_result)

        # Create main UI panels with optimized drawing
        self._draw_main_header_optimized(display_frame, current_time)
        self._draw_eye_tracking_panel_optimized(display_frame, frame_result, w, elapsed)
        self._draw_posture_panel_optimized(display_frame, frame_result, h, w)
        self._draw_health_status_panel_optimized(display_frame, frame_result, h, elapsed)
        self._draw_statistics_panel_optimized(display_frame, h, elapsed)
        self._draw_alerts_panel_optimized(display_frame, frame_result, h, w)

        # Draw timestamp
        self._draw_timestamp(display_frame)

        return display_frame

    def _draw_main_header_optimized(self, frame: np.ndarray, current_time: float):
        """
        Vẽ header chính với system info - Optimized version
        """
        h, w = frame.shape[:2]

        # Semi-transparent header background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.3, overlay, 0.7, 0, frame)

        # Main title
        cv2.putText(frame, "AEYEPRO HEALTH MONITORING SYSTEM", (15, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Session info
        cv2.putText(frame, f"Session: {self.session_id}", (15, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Calculate FPS only once per second
        self.display_fps_counter += 1
        if current_time - self.display_fps_time >= 1.0:
            display_fps = self.display_fps_counter / (current_time - self.display_fps_time)
            self.display_fps_counter = 0
            self.display_fps_time = current_time
            self.stats['fps'] = display_fps

        cv2.putText(frame, f"FPS: {self.stats['fps']:.1f} | Frames: {self.processed_frames} | Errors: {self.error_count}",
                   (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Status indicator
        status_color = (0, 255, 0) if self.error_count < 10 else (0, 255, 255) if self.error_count < 50 else (0, 0, 255)
        cv2.circle(frame, (w - 30, 30), 8, status_color, -1)
        cv2.putText(frame, "ACTIVE", (w - 100, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

    def _draw_eye_tracking_panel_optimized(self, frame: np.ndarray, frame_result: Dict[str, Any], frame_width: int, elapsed: float):
        """
        Vẽ eye tracking panel - Optimized version with pre-calculated elapsed time
        """
        eye_data = frame_result.get('eye_data', {})
        blink_data = frame_result.get('blink_data', {})

        # Panel dimensions
        panel_width = 280
        panel_height = 180
        panel_x = 15
        panel_y = 100

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                     (20, 20, 40), -1)
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)

        # Border
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                     (0, 200, 255), 2)

        # Title
        cv2.putText(frame, "EYE TRACKING", (panel_x + 10, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        # Separator line
        cv2.line(frame, (panel_x + 10, panel_y + 35), (panel_x + panel_width - 10, panel_y + 35),
                  (0, 200, 255), 1)

        y_offset = panel_y + 55
        line_height = 18

        # EAR metrics - batch processing for efficiency
        if eye_data.get('avg_ear'):
            ear = eye_data['avg_ear']
            left_ear = eye_data.get('left_ear', 0)
            right_ear = eye_data.get('right_ear', 0)

            # Color coding for EAR
            if ear > 0.27:
                ear_color = (0, 255, 0)
            elif ear > 0.22:
                ear_color = (0, 255, 255)
            else:
                ear_color = (0, 100, 255)

            # Draw EAR info
            cv2.putText(frame, f"L-EAR: {left_ear:.3f}", (panel_x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 255, 150), 1)
            y_offset += line_height

            cv2.putText(frame, f"R-EAR: {right_ear:.3f}", (panel_x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 255, 150), 1)
            y_offset += line_height

            cv2.putText(frame, f"AVG-EAR: {ear:.3f}", (panel_x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, ear_color, 2)
            y_offset += line_height + 5

            # EAR visual bar - simplified calculation
            bar_width = int(min((ear / 0.4) * 200, 200))
            cv2.rectangle(frame, (panel_x + 100, y_offset - 12), (panel_x + 100 + bar_width, y_offset - 8),
                         ear_color, -1)
            cv2.rectangle(frame, (panel_x + 100, y_offset - 12), (panel_x + 300, y_offset - 8),
                         (100, 100, 100), 1)
            y_offset += line_height

        # Blink rate - use pre-calculated elapsed time
        blink_rate = self.stats['total_blinks'] / max(elapsed/60, 0.1)

        if blink_data.get('blink_detected'):
            cv2.putText(frame, "STATUS: BLINK DETECTED!", (panel_x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "STATUS: NORMAL", (panel_x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        y_offset += line_height

        cv2.putText(frame, f"Total: {self.stats['total_blinks']} ({blink_rate:.1f}/min)",
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Distance
        if eye_data.get('distance_cm'):
            distance = eye_data['distance_cm']
            if 50 <= distance <= 80:
                distance_color = (0, 255, 0)
            elif 30 <= distance <= 100:
                distance_color = (255, 255, 0)
            else:
                distance_color = (255, 100, 100)
            cv2.putText(frame, f"Distance: {distance:.1f}cm", (panel_x + 10, y_offset + line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, distance_color, 1)

    def _draw_posture_panel_optimized(self, frame: np.ndarray, frame_result: Dict[str, Any], frame_height: int, frame_width: int):
        """
        Vẽ posture analysis panel - Optimized version
        """
        posture_data = frame_result.get('posture_data', {})

        # Panel dimensions - right side
        panel_width = 280
        panel_height = 200
        panel_x = frame_width - panel_width - 15
        panel_y = 100

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                     (40, 20, 20), -1)
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)

        # Border
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                     (255, 150, 0), 2)

        # Title
        cv2.putText(frame, "POSTURE ANALYSIS", (panel_x + 10, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)

        # Separator
        cv2.line(frame, (panel_x + 10, panel_y + 35), (panel_x + panel_width - 10, panel_y + 35),
                  (255, 150, 0), 1)

        y_offset = panel_y + 55
        line_height = 18

        # Head Movement Tracking
        head_side_angle = posture_data.get('head_side_angle')
        if head_side_angle is not None and head_side_angle != 0:
            # Color coding based on threshold
            abs_angle = abs(head_side_angle)
            if abs_angle <= 15:
                color = (0, 255, 0)
                status = "GOOD"
            elif abs_angle <= 20:
                color = (0, 255, 255)
                status = "WARN"
            else:
                color = (0, 100, 255)
                status = "POOR"

            direction = "LEFT" if head_side_angle < 0 else "RIGHT" if head_side_angle > 0 else "CENTER"
            cv2.putText(frame, f"Head Turn: {abs_angle:.1f}° {direction}",
                       (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(frame, status, (panel_x + 180, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_offset += line_height

        head_updown_angle = posture_data.get('head_updown_angle')
        if head_updown_angle is not None and head_updown_angle != 0:
            abs_angle = abs(head_updown_angle)
            if abs_angle <= 15:
                color = (0, 255, 0)
                status = "GOOD"
            elif abs_angle <= 22:
                color = (0, 255, 255)
                status = "WARN"
            else:
                color = (0, 100, 255)
                status = "POOR"

            direction = "DOWN" if head_updown_angle > 0 else "UP" if head_updown_angle < 0 else "LEVEL"
            cv2.putText(frame, f"Head Tilt: {abs_angle:.1f}° {direction}",
                       (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(frame, status, (panel_x + 180, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_offset += line_height

        # Shoulder Analysis
        shoulder_tilt = posture_data.get('shoulder_tilt')
        if shoulder_tilt is not None and shoulder_tilt != 0:
            abs_tilt = abs(shoulder_tilt)
            if abs_tilt <= 10:
                color = (0, 255, 0)
                status = "GOOD"
            elif abs_tilt <= 15:
                color = (0, 255, 255)
                status = "WARN"
            else:
                color = (0, 100, 255)
                status = "POOR"

            direction = "LEFT" if shoulder_tilt < 0 else "RIGHT" if shoulder_tilt > 0 else "LEVEL"
            cv2.putText(frame, f"Shoulder: {abs_tilt:.1f}° {direction}",
                       (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(frame, status, (panel_x + 180, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_offset += line_height + 5

        # Distance from camera
        distance = posture_data.get('eye_distance_cm') or posture_data.get('distance_cm')
        if distance is not None:
            if 50 <= distance <= 80:
                distance_color = (0, 255, 0)
                distance_status = "OPTIMAL"
            elif 30 <= distance <= 100:
                distance_color = (255, 255, 0)
                distance_status = "OK"
            else:
                distance_color = (255, 100, 100)
                distance_status = "POOR"

            cv2.putText(frame, f"Distance: {distance:.1f} cm",
                       (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, distance_color, 1)
            cv2.putText(frame, distance_status, (panel_x + 140, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, distance_color, 1)
            y_offset += line_height + 5

        # Visual posture indicator
        self._draw_posture_visual_indicator(frame, panel_x + 20, y_offset, posture_data)

        # Overall posture status
        y_offset += 60
        posture_status = posture_data.get('status', 'unknown')
        if posture_status == 'good':
            overall_color = (0, 255, 0)
            status_text = "[OK] GOOD POSTURE"
        elif posture_status == 'poor':
            overall_color = (0, 100, 255)
            status_text = "[WARN] POOR POSTURE"
        else:
            overall_color = (255, 255, 0)
            status_text = "[?] UNKNOWN"

        cv2.putText(frame, status_text, (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, overall_color, 2)

    def _draw_health_status_panel_optimized(self, frame: np.ndarray, frame_result: Dict[str, Any], frame_height: int, elapsed: float):
        """
        Vẽ health status panel - Optimized version with pre-calculated elapsed time
        """
        # Panel dimensions - left bottom
        panel_width = 350
        panel_height = 160
        panel_x = 15
        panel_y = frame_height - panel_height - 15

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                     (20, 40, 20), -1)
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)

        # Border
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                     (0, 255, 100), 2)

        # Title
        cv2.putText(frame, "HEALTH STATUS", (panel_x + 10, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)

        # Separator
        cv2.line(frame, (panel_x + 10, panel_y + 35), (panel_x + panel_width - 10, panel_y + 35),
                  (0, 255, 100), 1)

        y_offset = panel_y + 55
        line_height = 20

        # Get data
        eye_data = frame_result.get('eye_data', {})
        blink_data = frame_result.get('blink_data', {})
        drowsy_data = frame_result.get('drowsy_data', {})

        # Session duration - use pre-calculated elapsed time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        cv2.putText(frame, f"Session: {hours:02d}:{minutes:02d}:{seconds:02d}",
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += line_height

        # Blink rate analysis - use pre-calculated elapsed time
        blink_rate = self.stats['total_blinks'] / max(elapsed/60, 0.1)
        if 15 <= blink_rate <= 25:
            blink_health = "NORMAL"
            blink_color = (0, 255, 0)
        else:
            blink_health = "ABNORMAL"
            blink_color = (255, 255, 0)

        cv2.putText(frame, f"Blink Rate: {blink_rate:.1f}/min ({blink_health})",
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, blink_color, 1)
        y_offset += line_height

        # Drowsiness status
        if drowsy_data.get('drowsiness_detected'):
            drowsy_color = (0, 100, 255)
            drowsy_text = "[!] DROWSINESS DETECTED!"
            ear_duration = drowsy_data.get('ear_duration', 0)
            posture_duration = drowsy_data.get('posture_bad_duration', 0)
            gaze_duration = drowsy_data.get('gaze_off_duration', 0)
            reason = drowsy_data.get('reason', 'Unknown')

            # Show the most relevant duration
            if ear_duration > 0:
                duration_text = f"EAR Low: {ear_duration:.1f}s (3s threshold)"
            elif posture_duration > 0:
                duration_text = f"Bad Posture: {posture_duration:.1f}s"
            elif gaze_duration > 0:
                duration_text = f"Gaze Off: {gaze_duration:.1f}s"
            else:
                duration_text = f"Reason: {reason}"
        else:
            drowsy_color = (0, 255, 0)
            drowsy_text = "[OK] AWAKE & ALERT"
            duration_text = f"Events: {self.stats['drowsy_events']}"

        cv2.putText(frame, drowsy_text, (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, drowsy_color, 2)
        y_offset += line_height

        cv2.putText(frame, duration_text, (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, drowsy_color, 1)
        y_offset += line_height

        # Eye fatigue indicator
        if eye_data.get('avg_ear'):
            avg_ear = self.stats.get('avg_ear', eye_data['avg_ear'])
            if avg_ear < 0.2:
                fatigue_level = "HIGH"
                fatigue_color = (0, 100, 255)
            elif avg_ear < 0.25:
                fatigue_level = "MEDIUM"
                fatigue_color = (255, 255, 0)
            else:
                fatigue_level = "LOW"
                fatigue_color = (0, 255, 0)

            cv2.putText(frame, f"Eye Fatigue: {fatigue_level}",
                       (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fatigue_color, 1)

    def _draw_statistics_panel_optimized(self, frame: np.ndarray, frame_height: int, elapsed: float):
        """
        Vẽ statistics panel - Optimized version with pre-calculated elapsed time
        """
        # Panel dimensions - center bottom
        panel_width = 320
        panel_height = 120
        panel_x = 390  # Position after health status panel
        panel_y = frame_height - panel_height - 15

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                     (40, 40, 20), -1)
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)

        # Border
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                     (255, 200, 0), 2)

        # Title
        cv2.putText(frame, "REAL-TIME STATISTICS", (panel_x + 10, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        # Separator
        cv2.line(frame, (panel_x + 10, panel_y + 35), (panel_x + panel_width - 10, panel_y + 35),
                  (255, 200, 0), 1)

        y_offset = panel_y + 55
        line_height = 18

        # Performance metrics
        cv2.putText(frame, f"Processed: {self.processed_frames:,} frames",
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y_offset += line_height

        success_rate = (self.processed_frames / max(self.frame_count, 1)) * 100
        cv2.putText(frame, f"Success Rate: {success_rate:.1f}%",
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y_offset += line_height

        cv2.putText(frame, f"Posture Alerts: {self.stats['posture_alerts']}",
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y_offset += line_height

        # Progress bar for session - use pre-calculated elapsed time
        progress = min((elapsed / 3600) * 100, 100)  # Progress towards 1 hour
        bar_width = int((progress / 100) * 200)

        cv2.putText(frame, f"Session Progress: {progress:.1f}%",
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y_offset += line_height

        # Progress bar
        cv2.rectangle(frame, (panel_x + 10, y_offset), (panel_x + 10 + bar_width, y_offset + 10),
                     (0, 255, 0), -1)
        cv2.rectangle(frame, (panel_x + 10, y_offset), (panel_x + 210, y_offset + 10),
                     (100, 100, 100), 1)

    def _draw_alerts_panel_optimized(self, frame: np.ndarray, frame_result: Dict[str, Any], frame_height: int, frame_width: int):
        """
        Vẽ alerts panel - Optimized version
        """
        # Panel dimensions - right bottom
        panel_width = 280
        panel_height = 140
        panel_x = frame_width - panel_width - 15
        panel_y = frame_height - panel_height - 15

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                     (40, 20, 40), -1)
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)

        # Border (color based on alert level)
        eye_data = frame_result.get('eye_data', {})
        posture_data = frame_result.get('posture_data', {})
        drowsy_data = frame_result.get('drowsy_data', {})

        # Determine alert level
        alert_count = 0
        if eye_data.get('avg_ear', 0.3) < 0.22: alert_count += 1
        if posture_data.get('status') == 'poor': alert_count += 1
        if drowsy_data.get('drowsiness_detected'): alert_count += 2  # Higher priority

        if alert_count >= 3:
            border_color = (0, 100, 255)  # Red - Critical
        elif alert_count >= 2:
            border_color = (0, 255, 255)  # Yellow - Warning
        else:
            border_color = (0, 255, 0)    # Green - Good

        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                     border_color, 2)

        # Title
        cv2.putText(frame, "ALERTS & RECOMMENDATIONS", (panel_x + 10, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, border_color, 2)

        # Separator
        cv2.line(frame, (panel_x + 10, panel_y + 35), (panel_x + panel_width - 10, panel_y + 35),
                  border_color, 1)

        y_offset = panel_y + 55
        line_height = 16

        # Generate alerts
        alerts = []

        # Eye-related alerts
        if eye_data.get('distance_cm'):
            distance = eye_data['distance_cm']
            if distance < 40:
                alerts.append(("[DIST] Too close to screen!", (255, 100, 100)))
            elif distance > 100:
                alerts.append(("[DIST] Too far from screen!", (255, 100, 100)))

        if eye_data.get('avg_ear', 0.3) < 0.22:
            alerts.append(("[EYE] Eye fatigue detected", (255, 255, 0)))

        # Posture alerts
        if posture_data.get('status') == 'poor':
            alerts.append(("[POSTURE] Poor sitting posture", (255, 150, 0)))

        # Drowsiness alerts - check for EAR approaching threshold
        ear_value = eye_data.get('avg_ear', 0.3)
        if drowsy_data.get('drowsiness_detected'):
            alerts.append(("[!] DROWSINESS WARNING! (3s EAR low)", (0, 100, 255)))
        elif ear_value < 0.28:  # Warning threshold before 3s mark
            alerts.append(("[!] EAR getting low - stay alert!", (255, 255, 0)))

        # Blink rate alerts - use elapsed time from start
        elapsed = time.time() - self.start_time if self.start_time else 0
        blink_rate = self.stats['total_blinks'] / max(elapsed/60, 0.1)
        if blink_rate < 10:
            alerts.append(("[BLINK] Low blink rate - dry eyes", (255, 200, 0)))
        elif blink_rate > 30:
            alerts.append(("[BLINK] High blink rate - stress?", (255, 200, 0)))

        # Display alerts (max 4)
        for i, (alert_text, color) in enumerate(alerts[:4]):
            if y_offset + line_height < panel_y + panel_height - 10:
                cv2.putText(frame, alert_text, (panel_x + 10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                y_offset += line_height

        # If no alerts
        if not alerts:
            cv2.putText(frame, "[OK] All systems normal", (panel_x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def _draw_face_landmarks(self, frame: np.ndarray, frame_result: Dict[str, Any]):
        """
        Vẽ MediaPipe face landmarks và eye tracking visualization
        """
        eye_data = frame_result.get('eye_data', {})

        if eye_data and eye_data.get('landmarks'):
            # Draw eye regions with enhanced visualization
            if eye_data.get('left_eye') is not None and eye_data.get('right_eye') is not None:
                left_eye = eye_data['left_eye']
                right_eye = eye_data['right_eye']

                # Draw left eye with connected points
                for i, (x, y) in enumerate(left_eye):
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 100), -1)
                    if i < len(left_eye) - 1:
                        cv2.line(frame, (int(x), int(y)),
                               (int(left_eye[i+1][0]), int(left_eye[i+1][1])),
                               (0, 255, 150), 2)
                cv2.line(frame, (int(left_eye[-1][0]), int(left_eye[-1][1])),
                       (int(left_eye[0][0]), int(left_eye[0][1])),
                       (0, 255, 150), 2)

                # Draw right eye with connected points
                for i, (x, y) in enumerate(right_eye):
                    cv2.circle(frame, (int(x), int(y)), 3, (100, 255, 0), -1)
                    if i < len(right_eye) - 1:
                        cv2.line(frame, (int(x), int(y)),
                               (int(right_eye[i+1][0]), int(right_eye[i+1][1])),
                               (150, 255, 0), 2)
                cv2.line(frame, (int(right_eye[-1][0]), int(right_eye[-1][1])),
                       (int(right_eye[0][0]), int(right_eye[0][1])),
                       (150, 255, 0), 2)

            # Draw gaze point with enhanced visualization
            if eye_data.get('gaze_point'):
                gaze_x, gaze_y = eye_data['gaze_point']
                # Draw crosshair for gaze point
                cv2.circle(frame, (int(gaze_x), int(gaze_y)), 10, (0, 0, 255), -1)
                cv2.line(frame, (int(gaze_x), int(gaze_y - 15)),
                       (int(gaze_x), int(gaze_y + 15)), (0, 0, 255), 3)
                cv2.line(frame, (int(gaze_x - 15), int(gaze_y)),
                       (int(gaze_x + 15), int(gaze_y)), (0, 0, 255), 3)

                # Draw gaze circle
                cv2.circle(frame, (int(gaze_x), int(gaze_y)), 30, (0, 0, 255), 1)

    def _draw_main_header(self, frame: np.ndarray):
        """
        Vẽ header chính với system info
        """
        h, w = frame.shape[:2]

        # Semi-transparent header background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.3, overlay, 0.7, 0, frame)

        # Main title
        cv2.putText(frame, "AEYEPRO HEALTH MONITORING SYSTEM", (15, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Session info
        cv2.putText(frame, f"Session: {self.session_id}", (15, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Calculate FPS
        current_time = time.time()
        self.display_fps_counter += 1
        if current_time - self.display_fps_time >= 1.0:
            display_fps = self.display_fps_counter / (current_time - self.display_fps_time)
            self.display_fps_counter = 0
            self.display_fps_time = current_time
        else:
            display_fps = self.stats.get('fps', 0)

        cv2.putText(frame, f"FPS: {display_fps:.1f} | Frames: {self.processed_frames} | Errors: {self.error_count}",
                   (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Status indicator
        status_color = (0, 255, 0) if self.error_count < 10 else (0, 255, 255) if self.error_count < 50 else (0, 0, 255)
        cv2.circle(frame, (w - 30, 30), 8, status_color, -1)
        cv2.putText(frame, "ACTIVE", (w - 100, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

    def _draw_eye_tracking_panel(self, frame: np.ndarray, frame_result: Dict[str, Any], frame_width: int):
        """
        Vẽ eye tracking panel với comprehensive eye metrics
        """
        eye_data = frame_result.get('eye_data', {})
        blink_data = frame_result.get('blink_data', {})

        # Panel dimensions
        panel_width = 280
        panel_height = 180
        panel_x = 15
        panel_y = 100

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                     (20, 20, 40), -1)
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)

        # Border
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                     (0, 200, 255), 2)

        # Title
        cv2.putText(frame, "EYE TRACKING", (panel_x + 10, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        # Separator line
        cv2.line(frame, (panel_x + 10, panel_y + 35), (panel_x + panel_width - 10, panel_y + 35),
                  (0, 200, 255), 1)

        y_offset = panel_y + 55
        line_height = 18

        # EAR metrics
        if eye_data.get('avg_ear'):
            ear = eye_data['avg_ear']
            left_ear = eye_data.get('left_ear', 0)
            right_ear = eye_data.get('right_ear', 0)

            # Color coding for EAR
            ear_color = (0, 255, 0) if ear > 0.27 else (0, 255, 255) if ear > 0.22 else (0, 100, 255)

            cv2.putText(frame, f"L-EAR: {left_ear:.3f}", (panel_x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 255, 150), 1)
            y_offset += line_height

            cv2.putText(frame, f"R-EAR: {right_ear:.3f}", (panel_x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 255, 150), 1)
            y_offset += line_height

            cv2.putText(frame, f"AVG-EAR: {ear:.3f}", (panel_x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, ear_color, 2)
            y_offset += line_height + 5

            # EAR visual bar
            bar_width = int((ear / 0.4) * 200)  # Max EAR ~0.4
            cv2.rectangle(frame, (panel_x + 100, y_offset - 12), (panel_x + 100 + bar_width, y_offset - 8),
                         ear_color, -1)
            cv2.rectangle(frame, (panel_x + 100, y_offset - 12), (panel_x + 300, y_offset - 8),
                         (100, 100, 100), 1)
            y_offset += line_height

        # Eye contrast
        if eye_data.get('avg_contrast'):
            contrast = eye_data['avg_contrast']
            contrast_color = (0, 255, 0) if contrast > 15 else (255, 255, 0) if contrast > 8 else (255, 100, 100)
            cv2.putText(frame, f"Contrast: {contrast:.1f}", (panel_x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, contrast_color, 1)
            y_offset += line_height

        # Blink status and count
        blink_rate = self.stats['total_blinks'] / max((time.time() - self.start_time) / 60, 0.1)

        if blink_data.get('blink_detected'):
            cv2.putText(frame, "STATUS: BLINK DETECTED!", (panel_x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "STATUS: NORMAL", (panel_x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        y_offset += line_height

        cv2.putText(frame, f"Total: {self.stats['total_blinks']} ({blink_rate:.1f}/min)",
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Distance
        if eye_data.get('distance_cm'):
            distance = eye_data['distance_cm']
            distance_color = (0, 255, 0) if 50 <= distance <= 80 else (255, 255, 0) if 30 <= distance <= 100 else (255, 100, 100)
            cv2.putText(frame, f"Distance: {distance:.1f}cm", (panel_x + 10, y_offset + line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, distance_color, 1)

    def _draw_posture_panel(self, frame: np.ndarray, frame_result: Dict[str, Any], frame_height: int, frame_width: int):
        """
        Vẽ comprehensive posture analysis panel
        """
        posture_data = frame_result.get('posture_data', {})

        # Panel dimensions - right side
        panel_width = 280
        panel_height = 200
        panel_x = frame_width - panel_width - 15
        panel_y = 100

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                     (40, 20, 20), -1)
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)

        # Border
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                     (255, 150, 0), 2)

        # Title
        cv2.putText(frame, "POSTURE ANALYSIS", (panel_x + 10, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)

        # Separator
        cv2.line(frame, (panel_x + 10, panel_y + 35), (panel_x + panel_width - 10, panel_y + 35),
                  (255, 150, 0), 1)

        y_offset = panel_y + 55
        line_height = 18

        # Head Movement Tracking
        head_side_angle = posture_data.get('head_side_angle')
        if head_side_angle is not None:
            # Color coding based on threshold
            abs_angle = abs(head_side_angle)
            if abs_angle <= 15:
                color = (0, 255, 0)
                status = "GOOD"
            elif abs_angle <= 20:
                color = (0, 255, 255)
                status = "WARNING"
            else:
                color = (0, 100, 255)
                status = "POOR"

            direction = "LEFT" if head_side_angle < 0 else "RIGHT" if head_side_angle > 0 else "CENTER"
            cv2.putText(frame, f"Head Turn: {abs_angle:.1f}° {direction}",
                       (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            # Status indicator
            cv2.putText(frame, status, (panel_x + 180, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_offset += line_height

        head_updown_angle = posture_data.get('head_updown_angle')
        if head_updown_angle is not None:
            abs_angle = abs(head_updown_angle)
            if abs_angle <= 15:
                color = (0, 255, 0)
                status = "GOOD"
            elif abs_angle <= 22:
                color = (0, 255, 255)
                status = "WARNING"
            else:
                color = (0, 100, 255)
                status = "POOR"

            direction = "DOWN" if head_updown_angle > 0 else "UP" if head_updown_angle < 0 else "LEVEL"
            cv2.putText(frame, f"Head Tilt: {abs_angle:.1f}° {direction}",
                       (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(frame, status, (panel_x + 180, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_offset += line_height

        # Shoulder Analysis
        shoulder_tilt = posture_data.get('shoulder_tilt')
        if shoulder_tilt is not None:
            abs_tilt = abs(shoulder_tilt)
            if abs_tilt <= 10:
                color = (0, 255, 0)
                status = "GOOD"
            elif abs_tilt <= 15:
                color = (0, 255, 255)
                status = "WARNING"
            else:
                color = (0, 100, 255)
                status = "POOR"

            direction = "LEFT" if shoulder_tilt < 0 else "RIGHT" if shoulder_tilt > 0 else "LEVEL"
            cv2.putText(frame, f"Shoulder: {abs_tilt:.1f}° {direction}",
                       (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(frame, status, (panel_x + 180, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_offset += line_height + 5

        # Distance from camera
        distance = posture_data.get('eye_distance_cm') or posture_data.get('distance_cm')
        if distance is not None:
            if 50 <= distance <= 80:
                distance_color = (0, 255, 0)
                distance_status = "OPTIMAL"
            elif 30 <= distance <= 100:
                distance_color = (255, 255, 0)
                distance_status = "ACCEPTABLE"
            else:
                distance_color = (255, 100, 100)
                distance_status = "POOR"

            cv2.putText(frame, f"Distance: {distance:.1f} cm",
                       (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, distance_color, 1)
            cv2.putText(frame, distance_status, (panel_x + 140, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, distance_color, 1)
            y_offset += line_height + 5

        # Visual posture indicator
        self._draw_posture_visual_indicator(frame, panel_x + 20, y_offset, posture_data)

        # Overall posture status
        y_offset += 60
        posture_status = posture_data.get('status', 'unknown')
        if posture_status == 'good':
            overall_color = (0, 255, 0)
            status_text = "[OK] GOOD POSTURE"
        elif posture_status == 'poor':
            overall_color = (0, 100, 255)
            status_text = "[WARN] POOR POSTURE"
        else:
            overall_color = (255, 255, 0)
            status_text = "[?] UNKNOWN"

        cv2.putText(frame, status_text, (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, overall_color, 2)

    def _draw_posture_visual_indicator(self, frame: np.ndarray, x: int, y: int, posture_data: Dict[str, Any]):
        """
        Vẽ visual indicator cho posture (simplified human figure)
        """
        # Draw simple stick figure representation
        figure_height = 50
        figure_width = 30

        # Head
        head_center = (x + figure_width // 2, y + 10)
        cv2.circle(frame, head_center, 8, (255, 255, 255), 2)

        # Body
        body_start = (head_center[0], head_center[1] + 8)
        body_end = (head_center[0], y + 35)
        cv2.line(frame, body_start, body_end, (255, 255, 255), 2)

        # Arms
        shoulder_y = body_start[1] + 8
        left_arm_end = (body_start[0] - 15, shoulder_y + 15)
        right_arm_end = (body_start[0] + 15, shoulder_y + 15)

        # Color arms based on shoulder tilt
        shoulder_tilt = posture_data.get('shoulder_tilt', 0)
        if shoulder_tilt is None:
            shoulder_tilt = 0
        arm_color = (0, 255, 0) if abs(shoulder_tilt) <= 10 else (255, 255, 0) if abs(shoulder_tilt) <= 15 else (255, 100, 100)

        cv2.line(frame, body_start, left_arm_end, arm_color, 3)
        cv2.line(frame, body_start, right_arm_end, arm_color, 3)

        # Legs
        legs_start = body_end
        left_leg_end = (legs_start[0] - 10, legs_start[1] + 15)
        right_leg_end = (legs_start[0] + 10, legs_start[1] + 15)
        cv2.line(frame, legs_start, left_leg_end, (255, 255, 255), 2)
        cv2.line(frame, legs_start, right_leg_end, (255, 255, 255), 2)

    def _draw_health_status_panel(self, frame: np.ndarray, frame_result: Dict[str, Any], frame_height: int):
        """
        Vẽ comprehensive health status panel
        """
        # Panel dimensions - left bottom
        panel_width = 350
        panel_height = 160
        panel_x = 15
        panel_y = frame_height - panel_height - 15

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                     (20, 40, 20), -1)
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)

        # Border
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                     (0, 255, 100), 2)

        # Title
        cv2.putText(frame, "HEALTH STATUS", (panel_x + 10, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)

        # Separator
        cv2.line(frame, (panel_x + 10, panel_y + 35), (panel_x + panel_width - 10, panel_y + 35),
                  (0, 255, 100), 1)

        y_offset = panel_y + 55
        line_height = 20

        # Get data
        eye_data = frame_result.get('eye_data', {})
        blink_data = frame_result.get('blink_data', {})
        drowsy_data = frame_result.get('drowsy_data', {})

        # Session duration
        elapsed = time.time() - self.start_time if self.start_time else 0
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        cv2.putText(frame, f"Session: {hours:02d}:{minutes:02d}:{seconds:02d}",
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += line_height

        # Blink rate analysis
        blink_rate = self.stats['total_blinks'] / max(elapsed/60, 0.1)
        if 15 <= blink_rate <= 25:
            blink_health = "NORMAL"
            blink_color = (0, 255, 0)
        else:
            blink_health = "ABNORMAL"
            blink_color = (255, 255, 0)

        cv2.putText(frame, f"Blink Rate: {blink_rate:.1f}/min ({blink_health})",
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, blink_color, 1)
        y_offset += line_height

        # Drowsiness status
        if drowsy_data.get('drowsiness_detected'):
            drowsy_color = (0, 100, 255)
            drowsy_text = "[!] DROWSINESS DETECTED!"
            ear_duration = drowsy_data.get('ear_duration', 0)
            posture_duration = drowsy_data.get('posture_bad_duration', 0)
            gaze_duration = drowsy_data.get('gaze_off_duration', 0)
            reason = drowsy_data.get('reason', 'Unknown')

            # Show the most relevant duration
            if ear_duration > 0:
                duration_text = f"EAR Low: {ear_duration:.1f}s (3s threshold)"
            elif posture_duration > 0:
                duration_text = f"Bad Posture: {posture_duration:.1f}s"
            elif gaze_duration > 0:
                duration_text = f"Gaze Off: {gaze_duration:.1f}s"
            else:
                duration_text = f"Reason: {reason}"
        else:
            drowsy_color = (0, 255, 0)
            drowsy_text = "[OK] AWAKE & ALERT"
            duration_text = f"Events: {self.stats['drowsy_events']}"

        cv2.putText(frame, drowsy_text, (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, drowsy_color, 2)
        y_offset += line_height

        cv2.putText(frame, duration_text, (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, drowsy_color, 1)
        y_offset += line_height

        # Eye fatigue indicator
        if eye_data.get('avg_ear'):
            avg_ear = self.stats.get('avg_ear', eye_data['avg_ear'])
            if avg_ear < 0.2:
                fatigue_level = "HIGH"
                fatigue_color = (0, 100, 255)
            elif avg_ear < 0.25:
                fatigue_level = "MEDIUM"
                fatigue_color = (255, 255, 0)
            else:
                fatigue_level = "LOW"
                fatigue_color = (0, 255, 0)

            cv2.putText(frame, f"Eye Fatigue: {fatigue_level}",
                       (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fatigue_color, 1)

    def _draw_statistics_panel(self, frame: np.ndarray, frame_height: int):
        """
        Vẽ real-time statistics panel
        """
        # Panel dimensions - center bottom
        panel_width = 320
        panel_height = 120
        panel_x = 390  # Position after health status panel
        panel_y = frame_height - panel_height - 15

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                     (40, 40, 20), -1)
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)

        # Border
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                     (255, 200, 0), 2)

        # Title
        cv2.putText(frame, "REAL-TIME STATISTICS", (panel_x + 10, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        # Separator
        cv2.line(frame, (panel_x + 10, panel_y + 35), (panel_x + panel_width - 10, panel_y + 35),
                  (255, 200, 0), 1)

        y_offset = panel_y + 55
        line_height = 18

        # Performance metrics
        cv2.putText(frame, f"Processed: {self.processed_frames:,} frames",
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y_offset += line_height

        success_rate = (self.processed_frames / max(self.frame_count, 1)) * 100
        cv2.putText(frame, f"Success Rate: {success_rate:.1f}%",
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y_offset += line_height

        cv2.putText(frame, f"Posture Alerts: {self.stats['posture_alerts']}",
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y_offset += line_height

        # Progress bar for session
        elapsed = time.time() - self.start_time if self.start_time else 0
        progress = min((elapsed / 3600) * 100, 100)  # Progress towards 1 hour
        bar_width = int((progress / 100) * 200)

        cv2.putText(frame, f"Session Progress: {progress:.1f}%",
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y_offset += line_height

        # Progress bar
        cv2.rectangle(frame, (panel_x + 10, y_offset), (panel_x + 10 + bar_width, y_offset + 10),
                     (0, 255, 0), -1)
        cv2.rectangle(frame, (panel_x + 10, y_offset), (panel_x + 210, y_offset + 10),
                     (100, 100, 100), 1)

    def _draw_alerts_panel(self, frame: np.ndarray, frame_result: Dict[str, Any], frame_height: int, frame_width: int):
        """
        Vẽ alerts panel với real-time warnings và recommendations
        """
        # Panel dimensions - right bottom
        panel_width = 280
        panel_height = 140
        panel_x = frame_width - panel_width - 15
        panel_y = frame_height - panel_height - 15

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                     (40, 20, 40), -1)
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)

        # Border (color based on alert level)
        eye_data = frame_result.get('eye_data', {})
        posture_data = frame_result.get('posture_data', {})
        drowsy_data = frame_result.get('drowsy_data', {})

        # Determine alert level
        alert_count = 0
        if eye_data.get('avg_ear', 0.3) < 0.22: alert_count += 1
        if posture_data.get('status') == 'poor': alert_count += 1
        if drowsy_data.get('is_drowsy'): alert_count += 2  # Higher priority

        if alert_count >= 3:
            border_color = (0, 100, 255)  # Red - Critical
        elif alert_count >= 2:
            border_color = (0, 255, 255)  # Yellow - Warning
        else:
            border_color = (0, 255, 0)    # Green - Good

        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                     border_color, 2)

        # Title
        cv2.putText(frame, "ALERTS & RECOMMENDATIONS", (panel_x + 10, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, border_color, 2)

        # Separator
        cv2.line(frame, (panel_x + 10, panel_y + 35), (panel_x + panel_width - 10, panel_y + 35),
                  border_color, 1)

        y_offset = panel_y + 55
        line_height = 16

        # Generate alerts
        alerts = []

        # Eye-related alerts
        if eye_data.get('distance_cm'):
            distance = eye_data['distance_cm']
            if distance < 40:
                alerts.append(("[DIST] Too close to screen!", (255, 100, 100)))
            elif distance > 100:
                alerts.append(("[DIST] Too far from screen!", (255, 100, 100)))

        if eye_data.get('avg_ear', 0.3) < 0.22:
            alerts.append(("[EYE] Eye fatigue detected", (255, 255, 0)))

        # Posture alerts
        if posture_data.get('status') == 'poor':
            alerts.append(("[POSTURE] Poor sitting posture", (255, 150, 0)))

        # Drowsiness alerts - check for EAR approaching threshold
        ear_value = eye_data.get('avg_ear', 0.3)
        if drowsy_data.get('drowsiness_detected'):
            alerts.append(("[!] DROWSINESS WARNING! (3s EAR low)", (0, 100, 255)))
        elif ear_value < 0.28:  # Warning threshold before 3s mark
            alerts.append(("[!] EAR getting low - stay alert!", (255, 255, 0)))

        # Blink rate alerts
        elapsed = time.time() - self.start_time if self.start_time else 0
        blink_rate = self.stats['total_blinks'] / max(elapsed/60, 0.1)
        if blink_rate < 10:
            alerts.append(("[BLINK] Low blink rate - dry eyes", (255, 200, 0)))
        elif blink_rate > 30:
            alerts.append(("[BLINK] High blink rate - stress?", (255, 200, 0)))

        # Display alerts (max 4)
        for i, (alert_text, color) in enumerate(alerts[:4]):
            if y_offset + line_height < panel_y + panel_height - 10:
                cv2.putText(frame, alert_text, (panel_x + 10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                y_offset += line_height

        # If no alerts
        if not alerts:
            cv2.putText(frame, "[OK] All systems normal", (panel_x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def _draw_timestamp(self, frame: np.ndarray):
        """
        Vẽ real-time timestamp
        """
        timestamp = time.strftime("%H:%M:%S")
        h, w = frame.shape[:2]

        # Semi-transparent background for timestamp
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - 120, 90), (w - 10, 115), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)

        cv2.putText(frame, timestamp, (w - 110, 108),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _add_camera_overlay(self, frame: np.ndarray, frame_result: Dict[str, Any]):
        """
        Thêm thông tin overlay lên camera feed

        Args:
            frame: Frame để thêm overlay
            frame_result: Data để hiển thị
        """
        h, w = frame.shape[:2]

        # Create semi-transparent overlay for text background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h-150), (w-10, h-10), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Get data for display
        eye_data = frame_result.get('eye_data', {})
        blink_data = frame_result.get('blink_data', {})
        drowsy_data = frame_result.get('drowsy_data', {})
        posture_data = frame_result.get('posture_data', {})

        # Display health metrics
        y_offset = h - 140
        line_height = 20

        # Eye metrics
        if eye_data.get('avg_ear'):
            ear = eye_data['avg_ear']
            ear_color = (0, 255, 0) if ear > 0.25 else (0, 255, 255) if ear > 0.2 else (0, 0, 255)
            cv2.putText(frame, f"EAR: {ear:.3f}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, ear_color, 1)
            y_offset += line_height

        # Blink status
        if blink_data.get('blink_detected'):
            cv2.putText(frame, "BLINK: DETECTED", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += line_height

        # Drowsiness status
        if drowsy_data.get('is_drowsy'):
            cv2.putText(frame, "STATUS: DROWSY!", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            y_offset += line_height
        else:
            cv2.putText(frame, "STATUS: AWAKE", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += line_height

        # Distance
        if eye_data.get('distance_cm'):
            distance = eye_data['distance_cm']
            distance_color = (0, 255, 0) if 50 <= distance <= 80 else (255, 255, 0)
            cv2.putText(frame, f"Distance: {distance:.1f}cm", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, distance_color, 1)
            y_offset += line_height

        # Posture quality
        if posture_data.get('posture_quality'):
            posture = posture_data['posture_quality']
            posture_color = (0, 255, 0) if posture == 'good' else (0, 255, 255)
            cv2.putText(frame, f"Posture: {posture.upper()}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, posture_color, 1)

        return frame

    def _add_posture_data_overlay(self, frame: np.ndarray, posture_data: Dict[str, Any]):
        """
        Hiển thị thông số posture analysis lên camera frame

        Args:
            frame: Frame để thêm overlay
            posture_data: Dữ liệu từ posture_analyzer
        """
        h, w = frame.shape[:2]

        # Create overlay panel on right side
        panel_width = 250
        panel_x = w - panel_width - 10
        panel_y = 10
        panel_height = 200

        # Semi-transparent background for posture data
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                     (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)

        # Title
        cv2.putText(frame, "POSTURE ANALYSIS", (panel_x + 10, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Line separator
        cv2.line(frame, (panel_x + 10, panel_y + 35), (panel_x + panel_width - 10, panel_y + 35),
                  (255, 255, 255), 1)

        y_offset = panel_y + 55
        line_height = 18

        # Head Angles from PostureAnalyzer
        head_side_angle = posture_data.get('head_side_angle')
        if head_side_angle is not None:
            # Color code based on threshold (config: max_head_side_angle = 20)
            color = (0, 255, 0) if abs(head_side_angle) <= 15 else (0, 255, 255) if abs(head_side_angle) <= 20 else (0, 0, 255)
            direction = "LEFT" if head_side_angle < 0 else "RIGHT" if head_side_angle > 0 else "CENTER"
            cv2.putText(frame, f"Head Turn: {abs(head_side_angle):.1f}° {direction}",
                       (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += line_height

        head_updown_angle = posture_data.get('head_updown_angle')
        if head_updown_angle is not None:
            # Color code based on threshold (config: max_head_updown_angle = 22)
            color = (0, 255, 0) if abs(head_updown_angle) <= 15 else (0, 255, 255) if abs(head_updown_angle) <= 22 else (0, 0, 255)
            direction = "DOWN" if head_updown_angle > 0 else "UP" if head_updown_angle < 0 else "LEVEL"
            cv2.putText(frame, f"Head Tilt: {abs(head_updown_angle):.1f}° {direction}",
                       (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += line_height

        # Shoulder Tilt
        shoulder_tilt = posture_data.get('shoulder_tilt')
        if shoulder_tilt is not None:
            # Color code based on threshold (config: max_shoulder_tilt = 15)
            color = (0, 255, 0) if abs(shoulder_tilt) <= 10 else (0, 255, 255) if abs(shoulder_tilt) <= 15 else (0, 0, 255)
            direction = "LEFT" if shoulder_tilt < 0 else "RIGHT" if shoulder_tilt > 0 else "LEVEL"
            cv2.putText(frame, f"Shoulder Tilt: {abs(shoulder_tilt):.1f}° {direction}",
                       (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += line_height

        # Distance from camera
        eye_distance_cm = posture_data.get('eye_distance_cm')  # This is what PostureAnalyzer returns
        if eye_distance_cm is not None:
            # Color code based on reasonable range (50-80cm from config)
            color = (0, 255, 0) if 50 <= eye_distance_cm <= 80 else (255, 255, 0)
            cv2.putText(frame, f"Distance: {eye_distance_cm:.1f} cm",
                       (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += line_height

        # Posture Quality Status
        status = posture_data.get('status')
        if status:
            if status == 'good':
                color = (0, 255, 0)
                status_text = "GOOD POSTURE"
            elif status == 'poor':
                color = (0, 0, 255)
                status_text = "POOR POSTURE"
            else:
                color = (255, 255, 0)
                status_text = "UNKNOWN"

            cv2.putText(frame, status_text, (panel_x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Add timestamp
        import time
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(frame, f"Time: {timestamp}", (panel_x + 10, panel_y + panel_height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return frame

    def _normalize_posture_angles(self, posture_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Chuẩn hóa các góc về xung quanh 0 độ để cải thiện tính nhất quán

        Args:
            posture_data: Dict chứa các góc từ PostureAnalyzer

        Returns:
            Dict: posture_data với các góc đã chuẩn hóa
        """
        if not posture_data:
            return posture_data

        # Helper function to normalize angle to [-180, 180] range
        def normalize_angle(angle):
            if angle is None:
                return None
            # Normalize to [-180, 180]
            while angle > 180:
                angle -= 360
            while angle < -180:
                angle += 360
            return float(angle)

        # Normalize all angle values
        angle_keys = [
            'head_side_angle',    # Yaw: quay ngang
            'head_updown_angle',  # Pitch: nghiêng lên/xuống
            'head_roll',          # Roll: nghiêng bên
            'shoulder_tilt',      # Tilt vai
            'left_shoulder_angle',
            'right_shoulder_angle'
        ]

        for key in angle_keys:
            if key in posture_data and posture_data[key] is not None:
                posture_data[key] = normalize_angle(posture_data[key])

        return posture_data

    def update_fps_statistics(self):
        """
        Cập nhật FPS statistics cho display purposes
        """
        current_time = time.time()
        elapsed = current_time - self.start_time if self.start_time else 1
        self.stats['fps'] = self.processed_frames / elapsed if elapsed > 0 else 0

    def save_session_summary(self):
        """
        Save session summary to summary.csv - AEYE style
        """
        # Get comprehensive session statistics
        session_stats = self._collect_session_statistics()

        # Create summary row for summary.csv
        start_time = self.start_time or time.time()
        end_time = time.time()

        summary_row = {
            'session_id': self.session_id,
            'start_time': datetime.fromtimestamp(start_time).isoformat(),
            'end_time': datetime.fromtimestamp(end_time).isoformat(),
            'duration_minutes': round(session_stats['duration_minutes'], 2),
            # Health Metrics - chỉ quan trọng nhất
            'avg_ear': round(session_stats.get('avg_ear', 0), 4),
            'avg_distance_cm': round(session_stats.get('avg_distance_cm', 0), 2),
            'drowsiness_events': session_stats.get('drowsiness_events', 0),
            # Posture Analysis - 3 góc chính (correct field names)
            'avg_shoulder_tilt': round(session_stats.get('avg_shoulder_tilt_deg', 0), 2),
            'avg_head_pitch': round(session_stats.get('avg_head_pitch_deg', 0), 2),
            'avg_head_yaw': round(session_stats.get('avg_head_yaw_deg', 0), 2),
        }

        # ✅ SAVE TO SUMMARY CSV ONLY (không JSON - storage optimized)
        append_csv_row(summary_row, self.summary_csv_file)

        # ✅ HEALTH DATA COLLECTOR SẼ TỰ ĐỘNG QUẢN LÝ DATA LOGGING
        print("[INFO] Health Data Collector is handling all data storage automatically")
        print("[INFO] No JSON files created - optimized storage achieved")

        # Print summary to console - chỉ essential metrics
        duration_min = session_stats.get('duration_minutes', 0)
        print(f"\n[COMPLETED] Session {self.session_id} Summary:")
        print(f"  Duration: {duration_min:.1f} minutes")
        print(f"  Average EAR: {session_stats.get('avg_ear', 0):.3f}")
        print(f"  Average Distance: {session_stats.get('avg_distance_cm', 0):.1f} cm")
        print(f"  Drowsiness Events: {session_stats.get('drowsiness_events', 0)}")
        print(f"  Avg Shoulder Tilt: {session_stats.get('avg_shoulder_tilt_deg', 0):.1f}°")
        print(f"  Avg Head Pitch: {session_stats.get('avg_head_pitch_deg', 0):.1f}°")
        print(f"  Avg Head Yaw: {session_stats.get('avg_head_yaw_deg', 0):.1f}°")
        print(f"\nData saved to:")
        print(f"  Real-time data: Handled by Health Data Collector")
        print(f"  Summary: {self.summary_csv_file}")

    def run_application(self):
        """
        Chạy AEyePro Vision System application
        """
        print("AEyePro Vision System - Health Monitoring Application")
        print("=" * 60)

        # Initialize modules
        if not self.initialize_modules():
            return False

        # Setup session logging
        session_id = self.setup_session_logging()

        # ✅ START HEALTH DATA COLLECTOR (sử dụng đầy đủ tính năng)
        if self.health_collector:
            self.health_collector.start_collection()
            print(f"[OK] Health Data Collector started - Session: {session_id}")
            print(f"[INFO] Automatic data logging enabled with 3 key posture angles")
            print(f"[INFO] Storage optimized (50% reduction) - realtime CSV files")

        # Start application
        self.start_time = time.time()
        self.running = True

        print(f"\n[STARTING] Health monitoring session: {session_id}")
        print("All monitoring information will be displayed on camera overlay")
        print("Press Ctrl+C to stop monitoring or 'q' in camera window")
        print("=" * 60)

        try:
            while self.running and not self.shutdown_requested:
                loop_start = time.time()

                # Process frame
                frame_result = self.process_frame()

                # Update statistics
                self.update_statistics(frame_result)

                # Update FPS for display
                self.update_fps_statistics()

                # Save data
                self.save_frame_data(frame_result, self.frame_count)

                # Update camera display with comprehensive overlay
                self.display_camera_feed(frame_result)

                self.frame_count += 1

                # Frame rate control (target 30 FPS)
                processing_time = time.time() - loop_start
                target_frame_time = 1.0 / 30.0  # 30 FPS
                if processing_time < target_frame_time:
                    time.sleep(target_frame_time - processing_time)

            if self.shutdown_requested:
                print("\n[STOPPED] Monitoring stopped by user request")

        except Exception as e:
            print(f"\n[ERROR] Error during monitoring: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False
            self.shutdown()

        return True

    def shutdown(self):
        """
        Graceful shutdown của application
        """
        print("\n[SHUTTING DOWN] AEyePro Vision System...")

        # Stop all modules
        if self.eye_tracker:
            self.eye_tracker.stop()
            print("[OK] Eye tracker stopped")

        if self.health_collector:
            self.health_collector.stop_collection()
            print("[OK] Health Data Collector stopped - Session completed successfully")

            # ✅ DISPLAY STATISTICS FROM HEALTH DATA COLLECTOR
            try:
                collector_stats = self.health_collector.get_current_stats()
                print(f"[STATISTICS] Health Data Collector report:")
                print(f"  Total records: {collector_stats.get('total_records', 0)}")
                print(f"  Avg EAR: {collector_stats.get('avg_ear', 0):.3f}")
                print(f"  Avg distance: {collector_stats.get('avg_distance_cm', 0):.1f}cm")
                # Focus on 3 key posture angles
                print(f"  Avg shoulder tilt: {collector_stats.get('avg_shoulder_tilt_deg', 0):.1f}°")
                print(f"  Avg head pitch: {collector_stats.get('avg_head_pitch_deg', 0):.1f}°")
                print(f"  Avg head yaw: {collector_stats.get('avg_head_yaw_deg', 0):.1f}°")
            except Exception as e:
                print(f"[WARNING] Could not get statistics from Health Data Collector: {e}")

        # Close OpenCV windows
        if self.show_camera:
            cv2.destroyAllWindows()
            print("[OK] Camera windows closed")

        # Save session summary
        self.save_session_summary()

        print(f"[COMPLETED] Session {self.session_id} completed successfully!")
        print(f"[DATA SAVED] Data saved to: {self.session_dir}")
        print("[GOODBYE] Thank you for using AEyePro Vision System!")


def signal_handler(signum, frame):
    """Handler cho Ctrl+C - Graceful shutdown"""
    print("\n\n[SHUTDOWN] Signal received - Initiating graceful shutdown...")
    # Don't call sys.exit() here - let the application handle graceful shutdown


def main():
    """
    Main application entry point
    """
    parser = argparse.ArgumentParser(
        description='AEyePro Vision System - Real-time Health Monitoring',
        epilog='Example: python vision_app.py --config custom_settings.json'
    )
    parser.add_argument('--config', '-c', type=str, default='settings.json',
                       help='Configuration file path (default: settings.json)')
    parser.add_argument('--no-camera', action='store_true',
                       help='Disable camera display window (console only)')
    parser.add_argument('--version', '-v', action='version', version='AEyePro Vision System 3.0.0 - HOÀN CHỈNH')

    args = parser.parse_args()

    # Create application first
    app = AEyeProVisionApp(config_file=args.config, show_camera=not args.no_camera)

    # Setup signal handler with app reference
    def graceful_signal_handler(signum, frame):
        print("\n\n[SHUTDOWN] Signal received - Initiating graceful shutdown...")
        app.shutdown_requested = True

    signal.signal(signal.SIGINT, graceful_signal_handler)

    try:
        print("🚀 AEyePro Vision System v3.0.0 - HOÀN CHỈNH SỬ DỤNG TOÀN BỘ PACKAGE VISION")
        print("=" * 80)
        print("✅ SỬ DỤNG ĐẦY ĐỤ PACKAGE VISION:")
        print("   📦 Eye Tracker (468 landmarks) - Theo dõi mắt chính xác")
        print("   📦 Posture Analyzer (33 landmarks) - Phân tích tư thế")
        print("   📦 Blink Detector (EAR-based) - Detect chớp mắt real-time")
        print("   📦 Drowsiness Detector (multi-signal) - Phát hiện buồn ngủ")
        print("   📦 Health Data Collector (tự động) - Logging data tối ưu")
        print("🎯 TẬP TRUNG: 3 góc tư thế chính + storage tối ưu 50%")
        print("📊 Tất cả thông số hiển thị trên camera overlay")
        print("=" * 80)

        success = app.run_application()
        return 0 if success else 1
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)