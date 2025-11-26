# AEyePro - Computer Vision Health Monitoring System

## 📋 Giới Thiệu

AEyePro là hệ thống theo dõi sức khỏe sử dụng Computer Vision, tập trung vào 3 góc tư thế chính:

- **Eye Tracking**: Theo dõi mắt với MediaPipe Face Mesh (468 landmarks)
- **Posture Analysis**: Phân tích tư thế ngồi với 3 góc chính (vai, đầu trước-sau, đầu trái-phải)
- **Blink Detection**: Phát hiện và phân tích chớp mắt dựa trên EAR
- **Drowsiness Detection**: Phát hiện buồn ngủ và mệt mỏi
- **Health Data Collection**: Thu thập và lưu trữ dữ liệu sức khỏe (tối ưu 50% storage)

## 🚀 Cài Đặt

### 1. Yêu Cầu Hệ Thống
- Python 3.8+
- Camera webcam

### 2. Cài Đặt Dependencies
```bash
pip install -r requirements.txt
```

### 3. Kiểm Tra Cài Đặt
```bash
python main.py
```

## 🎮 Sử Dụng

### Chạy Console Version
```bash
cd AEyePro
python vision_app.py
```

---

# 🔧 GUI Integration Guide

## 📱 Build Custom GUI với PyQt6/CustomTkinter

Hướng dẫn này giúp bạn xây dựng giao diện đồ họa tùy chỉnh sử dụng core modules của AEyePro.

## 🏗️ Architecture Overview

```
GUI Application
├── Your UI (PyQt6/CustomTkinter)
├── AEyePro Core Modules
│   ├── AEyeProVisionApp (main class)
│   ├── EyeTracker
│   ├── PostureAnalyzer
│   ├── BlinkDetector
│   └── HealthDataCollector
└── Data Storage (CSV files)
```

## 🔑 Core Classes and Methods

### 1. AEyeProVisionApp (Main Class)

```python
from vision_app import AEyeProVisionApp

# Initialize ứng dụng
app = AEyeProVisionApp(
    config_file="settings.json",
    show_camera=False  # Disable OpenCV display for GUI
)
```

#### **Essential Methods:**

```python
# Khởi tạo modules
success = app.initialize_modules()
if not success:
    print("Failed to initialize modules")
    return

# Thiết lập logging
session_id = app.setup_session_logging()

# Xử lý một frame
frame_result = app.process_frame()

# Cập nhật thống kê
app.update_statistics(frame_result)

# Lưu data (gọi mỗi giây)
app.save_frame_data(frame_result, frame_count)

# Lấy dữ liệu realtime
latest_eye_data = app.eye_tracker.get_latest()
latest_posture_data = app.posture_analyzer.get_latest()
```

### 2. Eye Tracking Data

```python
# Lấy eye tracking data
eye_data = app.eye_tracker.get_latest()

if eye_data:
    print(f"EAR: {eye_data['avg_ear']:.3f}")
    print(f"Distance: {eye_data['distance_cm']:.1f}cm")
    print(f"Blinks: {eye_data.get('blink_count', 0)}")
    print(f"Left Eye EAR: {eye_data.get('left_ear', 0):.3f}")
    print(f"Right Eye EAR: {eye_data.get('right_ear', 0):.3f}")
```

### 3. Posture Analysis Data

```python
# Lấy posture analysis data
posture_data = app.posture_analyzer.get_latest()

if posture_data:
    print(f"Shoulder Tilt: {posture_data.get('shoulder_tilt', 0):.1f}°")
    print(f"Head Pitch (trước-sau): {posture_data.get('head_updown_angle', 0):.1f}°")
    print(f"Head Yaw (trái-phải): {posture_data.get('head_side_angle', 0):.1f}°")
    print(f"Posture Status: {posture_data.get('status', 'unknown')}")
```

### 4. Blink Detection Data

```python
# Lấy blink detection data
blink_data = app.blink_detector.update()

print(f"Blink Detected: {blink_data.get('blink_detected', False)}")
print(f"Blink Rate: {app.stats['total_blinks'] / max(elapsed/60, 0.1):.1f}/min")
```

### 5. Drowsiness Detection Data

```python
# Lấy drowsiness detection data
drowsy_data = app.drowsiness_detector.update(
    ear=eye_data.get('avg_ear'),
    posture_data=posture_data
)

print(f"Drowsiness Detected: {drowsy_data.get('drowsiness_detected', False)}")
print(f"Reason: {drowsy_data.get('reason', 'Unknown')}")
print(f"EAR Low Duration: {drowsy_data.get('ear_duration', 0):.1f}s")
```

## 🎨 PyQt6 Integration Example

```python
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtGui import QFont
import cv2
import numpy as np
from vision_app import AEyeProVisionApp

class AEyeProGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AEyePro - Health Monitoring")
        self.setGeometry(100, 100, 800, 600)

        # Initialize AEyePro
        self.aeye_app = AEyeProVisionApp(show_camera=False)

        # Setup UI
        self.setup_ui()

        # Setup timer for real-time updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_health_data)
        self.timer.start(33)  # ~30 FPS

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()

        # Health metrics labels
        self.ear_label = QLabel("EAR: --")
        self.distance_label = QLabel("Distance: --")
        self.blink_rate_label = QLabel("Blink Rate: --")
        self.posture_status_label = QLabel("Posture: --")
        self.drowsiness_label = QLabel("Status: --")

        # Posture angles labels (3 key angles)
        self.shoulder_tilt_label = QLabel("Shoulder Tilt: --")
        self.head_pitch_label = QLabel("Head Pitch: --")
        self.head_yaw_label = QLabel("Head Yaw: --")

        # Camera feed label
        self.camera_label = QLabel("Camera Feed")
        self.camera_label.setMinimumSize(640, 480)

        # Add to layout
        layout.addWidget(self.ear_label)
        layout.addWidget(self.distance_label)
        layout.addWidget(self.blink_rate_label)
        layout.addWidget(QLabel("\n--- POSTURE ANALYSIS ---"))
        layout.addWidget(self.shoulder_tilt_label)
        layout.addWidget(self.head_pitch_label)
        layout.addWidget(self.head_yaw_label)
        layout.addWidget(self.posture_status_label)
        layout.addWidget(QLabel("\n--- ALERT STATUS ---"))
        layout.addWidget(self.drowsiness_label)
        layout.addWidget(self.camera_label)

        central_widget.setLayout(layout)

        # Initialize AEyePro modules
        if self.aeye_app.initialize_modules():
            self.aeye_app.setup_session_logging()
            print("AEyePro initialized successfully")
        else:
            print("Failed to initialize AEyePro")

    def update_health_data(self):
        """Update health data every frame"""
        try:
            # Process frame
            frame_result = self.aeye_app.process_frame()

            if 'error' not in frame_result:
                # Update statistics
                self.aeye_app.update_statistics(frame_result)

                # Save data (every second - controlled internally)
                self.aeye_app.save_frame_data(frame_result, 0)

                # Update UI with latest data
                self.update_ui_display(frame_result)

                # Update camera feed
                self.update_camera_feed()

        except Exception as e:
            print(f"Error updating health data: {e}")

    def update_ui_display(self, frame_result):
        """Update UI labels with latest data"""
        # Get latest data
        eye_data = frame_result.get('eye_data', {})
        posture_data = frame_result.get('posture_data', {})
        drowsy_data = frame_result.get('drowsy_data', {})

        # Update eye metrics
        if eye_data:
            ear = eye_data.get('avg_ear', 0)
            distance = eye_data.get('distance_cm', 0)

            # Color coding for EAR
            ear_color = "green" if ear > 0.25 else "orange" if ear > 0.2 else "red"

            self.ear_label.setText(f"EAR: <span style='color: {ear_color}'>{ear:.3f}</span>")
            self.distance_label.setText(f"Distance: {distance:.1f}cm")

        # Update blink rate
        elapsed = time.time() - self.aeye_app.start_time if self.aeye_app.start_time else 1
        blink_rate = self.aeye_app.stats['total_blinks'] / max(elapsed/60, 0.1)
        self.blink_rate_label.setText(f"Blink Rate: {blink_rate:.1f}/min")

        # Update posture (3 key angles)
        if posture_data:
            shoulder_tilt = posture_data.get('shoulder_tilt', 0)
            head_pitch = posture_data.get('head_updown_angle', 0)
            head_yaw = posture_data.get('head_side_angle', 0)
            posture_status = posture_data.get('status', 'unknown')

            # Color coding for posture
            posture_color = "green" if posture_status == 'good' else "orange" if posture_status == 'unknown' else "red"

            self.shoulder_tilt_label.setText(f"Shoulder Tilt: {shoulder_tilt:.1f}°")
            self.head_pitch_label.setText(f"Head Pitch: {head_pitch:.1f}°")
            self.head_yaw_label.setText(f"Head Yaw: {head_yaw:.1f}°")
            self.posture_status_label.setText(f"Posture: <span style='color: {posture_color}'>{posture_status.upper()}</span>")

        # Update drowsiness status
        if drowsy_data.get('drowsiness_detected'):
            self.drowsiness_label.setText("<span style='color: red'>⚠ DROWSINESS DETECTED!</span>")
        else:
            self.drowsiness_label.setText("<span style='color: green'>✅ AWAKE & ALERT</span>")

    def update_camera_feed(self):
        """Update camera display"""
        try:
            frame = self.aeye_app.eye_tracker.get_frame()
            if frame is not None:
                # Convert frame to RGB for PyQt6
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)

                # Scale and display
                pixmap = QtGui.QPixmap.fromImage(qt_image)
                scaled_pixmap = pixmap.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
                self.camera_label.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"Error updating camera feed: {e}")

    def closeEvent(self, event):
        """Handle application close"""
        self.aeye_app.shutdown()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = AEyeProGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
```

## 🎨 CustomTkinter Integration Example

```python
import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk
from vision_app import AEyeProVisionApp

class AEyeProCustomTkinterApp:
    def __init__(self):
        # Setup CustomTkinter
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("AEyePro - Health Monitoring")
        self.root.geometry("1000x700")

        # Initialize AEyePro
        self.aeye_app = AEyeProVisionApp(show_camera=False)

        # Setup UI
        self.setup_ui()

        # Initialize AEyePro
        if self.aeye_app.initialize_modules():
            self.aeye_app.setup_session_logging()
            print("AEyePro initialized successfully")

            # Start update loop
            self.update_loop()
        else:
            print("Failed to initialize AEyePro")

    def setup_ui(self):
        # Main container
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Left panel - Health metrics
        left_panel = ctk.CTkFrame(main_frame)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 5))

        # Title
        title_label = ctk.CTkLabel(left_panel, text="AEyePro Health Monitor",
                                   font=ctk.CTkFont(size=20, weight="bold"))
        title_label.pack(pady=10)

        # Eye health frame
        eye_frame = ctk.CTkFrame(left_panel)
        eye_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(eye_frame, text="👁️ EYE HEALTH", font=ctk.CTkFont(size=16, weight="bold")).pack()

        self.ear_label = ctk.CTkLabel(eye_frame, text="EAR: --", font=ctk.CTkFont(size=14))
        self.ear_label.pack()

        self.blink_rate_label = ctk.CTkLabel(eye_frame, text="Blink Rate: --", font=ctk.CTkFont(size=14))
        self.blink_rate_label.pack()

        # Posture frame (3 key angles)
        posture_frame = ctk.CTkFrame(left_panel)
        posture_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(posture_frame, text="🪑 POSTURE ANALYSIS", font=ctk.CTkFont(size=16, weight="bold")).pack()

        self.shoulder_tilt_label = ctk.CTkLabel(posture_frame, text="Shoulder Tilt: --", font=ctk.CTkFont(size=14))
        self.shoulder_tilt_label.pack()

        self.head_pitch_label = ctk.CTkLabel(posture_frame, text="Head Pitch: --", font=ctk.CTkFont(size=14))
        self.head_pitch_label.pack()

        self.head_yaw_label = ctk.CTkLabel(posture_frame, text="Head Yaw: --", font=ctk.CTkFont(size=14))
        self.head_yaw_label.pack()

        self.posture_status_label = ctk.CTkLabel(posture_frame, text="Posture: --", font=ctk.CTkFont(size=14))
        self.posture_status_label.pack()

        # Status frame
        status_frame = ctk.CTkFrame(left_panel)
        status_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(status_frame, text="📊 STATUS", font=ctk.CTkFont(size=16, weight="bold")).pack()

        self.drowsiness_label = ctk.CTkLabel(status_frame, text="Status: Initializing...", font=ctk.CTkFont(size=14))
        self.drowsiness_label.pack()

        # Right panel - Camera feed
        right_panel = ctk.CTkFrame(main_frame)
        right_panel.pack(side="right", fill="both", expand=True, padx=(5, 0))

        ctk.CTkLabel(right_panel, text="📷 Camera Feed", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)

        self.camera_label = ctk.CTkLabel(right_panel, text="Camera initializing...")
        self.camera_label.pack(expand=True, fill="both", padx=10, pady=5)

    def update_loop(self):
        """Main update loop"""
        try:
            # Process frame
            frame_result = self.aeye_app.process_frame()

            if 'error' not in frame_result:
                # Update statistics
                self.aeye_app.update_statistics(frame_result)

                # Save data
                self.aeye_app.save_frame_data(frame_result, 0)

                # Update UI
                self.update_health_display(frame_result)
                self.update_camera_display()

        except Exception as e:
            print(f"Update loop error: {e}")

        # Schedule next update (30 FPS)
        self.root.after(33, self.update_loop)

    def update_health_display(self, frame_result):
        """Update health metrics display"""
        eye_data = frame_result.get('eye_data', {})
        posture_data = frame_result.get('posture_data', {})
        drowsy_data = frame_result.get('drowsy_data', {})

        # Update eye metrics
        if eye_data:
            ear = eye_data.get('avg_ear', 0)
            distance = eye_data.get('distance_cm', 0)

            # Color coding
            ear_text = f"EAR: {ear:.3f}"
            if ear > 0.25:
                self.ear_label.configure(text=ear_text, text_color=("green", "white"))
            elif ear > 0.2:
                self.ear_label.configure(text=ear_text, text_color=("orange", "white"))
            else:
                self.ear_label.configure(text=ear_text, text_color=("red", "white"))

            elapsed = time.time() - self.aeye_app.start_time if self.aeye_app.start_time else 1
            blink_rate = self.aeye_app.stats['total_blinks'] / max(elapsed/60, 0.1)
            self.blink_rate_label.configure(text=f"Blink Rate: {blink_rate:.1f}/min")

        # Update posture (3 key angles)
        if posture_data:
            shoulder_tilt = posture_data.get('shoulder_tilt', 0)
            head_pitch = posture_data.get('head_updown_angle', 0)
            head_yaw = posture_data.get('head_side_angle', 0)
            posture_status = posture_data.get('status', 'unknown')

            self.shoulder_tilt_label.configure(text=f"Shoulder Tilt: {shoulder_tilt:.1f}°")
            self.head_pitch_label.configure(text=f"Head Pitch: {head_pitch:.1f}°")
            self.head_yaw_label.configure(text=f"Head Yaw: {head_yaw:.1f}°")

            # Posture status color
            if posture_status == 'good':
                self.posture_status_label.configure(text="Posture: GOOD", text_color=("green", "white"))
            elif posture_status == 'poor':
                self.posture_status_label.configure(text="Posture: POOR", text_color=("red", "white"))
            else:
                self.posture_status_label.configure(text="Posture: UNKNOWN", text_color=("orange", "white"))

        # Update drowsiness
        if drowsy_data.get('drowsiness_detected'):
            self.drowsiness_label.configure(text="⚠ DROWSINESS DETECTED!", text_color=("red", "white"))
        else:
            self.drowsiness_label.configure(text="✅ AWAKE & ALERT", text_color=("green", "white"))

    def update_camera_display(self):
        """Update camera feed"""
        try:
            frame = self.aeye_app.eye_tracker.get_frame()
            if frame is not None:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Convert to PIL Image
                image = Image.fromarray(frame_rgb)

                # Convert to ImageTk
                image_tk = ImageTk.PhotoImage(image)

                # Update label
                self.camera_label.configure(image=image_tk)
                self.camera_label.image = image_tk  # Keep a reference

        except Exception as e:
            print(f"Camera display error: {e}")

    def run(self):
        """Run the application"""
        try:
            self.root.mainloop()
        finally:
            self.aeye_app.shutdown()

def main():
    app = AEyeProCustomTkinterApp()
    app.run()

if __name__ == "__main__":
    main()
```

## 🔧 Key Integration Points

### 1. Camera Access
```python
# Get raw camera frame
frame = app.eye_tracker.get_frame()

# Get processed frame with landmarks
eye_data = app.eye_tracker.get_latest()
```

### 2. Real-time Data Updates
```python
# Process frame (call this in your GUI timer)
frame_result = app.process_frame()

# Update statistics
app.update_statistics(frame_result)

# Save data (automatic with 1-second intervals)
app.save_frame_data(frame_result, frame_count)
```

### 3. Health Metrics Access
```python
# Get current statistics
stats = app.stats

# Get latest posture data
posture_data = app.posture_analyzer.get_latest()

# Get latest eye data
eye_data = app.eye_tracker.get_latest()

# Get blink data
blink_data = app.blink_detector.update()

# Get drowsiness data
drowsy_data = app.drowsiness_detector.update(
    ear=eye_data.get('avg_ear'),
    posture_data=posture_data
)
```

### 4. Session Management
```python
# Start new session
session_id = app.setup_session_logging()

# Save session summary (called automatically on shutdown)
app.save_session_summary()

# Graceful shutdown
app.shutdown()
```

## 📊 Data Output

### CSV Files Generated:
- `health_YYYYMMDD_HHMMSS.csv` - Real-time health data (optimized 9 fields)
- `summary.csv` - Session summaries with 3 key posture angles

### Key Data Fields:
- `timestamp`, `session_id`, `time_elapsed`
- `avg_ear`, `blink_count`, `blink_rate`
- `distance_cm`
- `shoulder_tilt`, `head_pitch`, `head_yaw` (3 key angles)
- `posture_status`, `drowsiness_detected`, `eye_fatigue_level`

## 🚀 Getting Started

1. **Install dependencies**:
```bash
pip install -r requirements.txt
pip install PyQt6  # hoặc pip install customtkinter
```

2. **Copy the example code** into your project

3. **Run the GUI**:
```bash
python your_gui_app.py
```

4. **Customize** the UI according to your needs

## 🎯 Best Practices

1. **Performance**: Use QTimer with ~33ms interval (30 FPS)
2. **Error Handling**: Wrap AEyePro calls in try-except blocks
3. **Memory Management**: Call `app.shutdown()` when closing
4. **Data Updates**: Don't call `save_frame_data()` manually (handled internally)
5. **Camera**: Don't enable OpenCV display (`show_camera=False`) when using GUI

## 📞 Troubleshooting

- **Camera issues**: Check `camera_index` in `config/settings.json`
- **Import errors**: Ensure all dependencies are installed
- **Performance**: Reduce frame rate in config if GUI is laggy
- **Memory usage**: Call `app.shutdown()` properly on exit

## 🔗 Additional Resources

- [MediaPipe Documentation](https://google.github.io/mediapipe/)
- [PyQt6 Documentation](https://www.riverbankcomputing.com/static/Docs/PyQt6/)
- [CustomTkinter Documentation](https://customtkinter.tomschimansky.com/)