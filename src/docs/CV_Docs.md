# AEyePro - Computer Vision Health Monitoring System

## 📋 Giới Thiệu

AEyePro là hệ thống theo dõi sức khỏe sử dụng Computer Vision với các tính năng chính:
- **Eye Tracking**: Theo dõi mắt với MediaPipe Face Mesh (468 landmarks)
- **Posture Analysis**: Phân tích tư thế ngồi với 3 góc chính (vai, đầu trước-sau, đầu trái-phải)
- **Blink Detection**: Phát hiện và phân tích chớp mắt dựa trên EAR
- **Drowsiness Detection**: Phát hiện buồn ngủ và mệt mỏi
- **Health Data Collection**: Thu thập và lưu trữ dữ liệu sức khỏe (tối ưu 50% storage)

## 🚀 Cài Đặt

### Yêu Cầu Hệ Thống
- Python 3.8+
- Camera webcam

### Cài Đặt Dependencies
```bash
pip install -r requirements.txt
```

### Kiểm Tra Cài Đặt
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

# 🔧 GUI Integration API

## 📱 Build Custom GUI với PyQt6/CustomTkinter

Dưới đây là danh sách các hàm cần thiết để tích hợp AEyePro vào giao diện đồ họa.

## 🏗️ Architecture Overview

```
GUI Application
├── Your UI (PyQt6/CustomTkinter)
├── AEyePro Core Modules
│   ├── AEyeProVisionApp (main class)
│   ├── EyeTracker
│   ├── PostureAnalyzer
│   ├── BlinkDetector
│   ├── DrowsinessDetector
│   └── HealthDataCollector
└── Data Storage (CSV files)
```

---

# Module 1: Eye Tracker (`vision/eye_tracker.py`)

## Class: `EyeTracker`
**Mục đích**: Theo dõi và phân tích đặc điểm mắt sử dụng MediaPipe Face Mesh (468 landmarks)

### Initialization Methods
```python
def __init__(self, config_path: str | Path = "settings.json")
```
- **Chức năng**: Khởi tạo MediaPipe Face Mesh, thiết lập camera
- **Tham số**: `config_path` - Đường dẫn file cấu hình
- **GUI Usage**: Gọi một lần khi khởi tạo ứng dụng

### Control Methods
```python
def start() -> None
```
- **Chức năng**: Bắt đầu camera và thread xử lý eye tracking
- **GUI Usage**: Gọi để bắt đầu capturing và processing video frames

```python
def stop() -> None
```
- **Chức năng**: Dừng tracking và giải phóng resources camera
- **GUI Usage**: Gọi khi dừng monitoring hoặc shutdown ứng dụng

### Data Access Methods
```python
def get_frame() -> Optional[np.ndarray]
```
- **Chức năng**: Lấy frame camera hiện tại từ buffer (thread-safe)
- **GUI Usage**: Sử dụng để hiển thị video feed trong GUI

```python
def get_latest() -> Dict[str, Any]
```
- **Chức năng**: Lấy kết quả processing mới nhất với thread safety
- **Trả về**: Dictionary với các keys:
  - `frame`: Frame đã xử lý
  - `left_eye`, `right_eye`: Tọa độ 6 điểm mắt
  - `gaze_point`: Điểm nhìn trên màn hình (x, y)
  - `left_ear`, `right_ear`: Eye Aspect Ratio cho mỗi mắt
  - `avg_ear`: EAR trung bình
  - `distance_cm`: Khoảng cách đến camera
- **GUI Usage**: Gọi thường xuyên để lấy data eye tracking cho display

### Calibration Method
```python
def calibrate_ear_thresholds(self, calibration_duration: float = 10.0) -> dict[str, Any]
```
- **Chức năng**: Calibrate tự động ngưỡng EAR cho người dùng cụ thể
- **Tham số**: `calibration_duration` - Thời gian calibrate (giây)
- **GUI Usage**: Gọi khi user muốn cá nhân hóa settings

---

# Module 2: Posture Analyzer (`vision/posture_analyzer.py`)

## Class: `PostureAnalyzer`
**Mục đích**: Phân tích tư thế ngồi sử dụng MediaPipe Pose (33 landmarks)

### Initialization Methods
```python
def __init__(self, config_path: str = "settings.json")
```
- **Chức năng**: Khởi tạo MediaPipe Pose với 33 landmarks, thiết lập filters
- **GUI Usage**: Gọi một lần khi khởi tạo ứng dụng

### Analysis Methods
```python
def analyze(self, frame: np.ndarray) -> Dict[str, Any]
```
- **Chức năng**: Phân tích tư thế từ frame video đầu vào
- **Tham số**: `frame` - Input image frame
- **Trả về**: Dictionary với các keys:
  - `head_side_angle`: Góc quay ngang đầu (-180 đến 180)
  - `head_updown_angle`: Góc nghiêng lên/xuống (-180 đến 180)
  - `shoulder_tilt`: Góc nghiêng vai (-180 đến 180)
  - `eye_distance_cm`: Khoảng cách ước tính đến camera
  - `status`: 'good', 'poor', hoặc 'unknown'
- **GUI Usage**: Gọi với frame hiện tại để lấy data posture

```python
def get_latest() -> Dict[str, Any]
```
- **Chức năng**: Lấy kết quả phân tích posture mới nhất
- **GUI Usage**: Gọi để lấy current posture status cho display

### Cleanup Methods
```python
def close() -> None
```
- **Chức năng**: Đóng MediaPipe resources và cleanup
- **GUI Usage**: Gọi khi shutdown ứng dụng

---

# Module 3: Blink Detector (`vision/blink_detector.py`)

## Class: `BlinkDetector`
**Mục đích**: Phát hiện và phân tích chớp mắt sử dụng thuật toán EAR-based

### Initialization Methods
```python
def __init__(self, config_path: str = "settings.json", eye_tracker: Optional[EyeTracker] = None)
```
- **Chức năng**: Khởi tạo blink detection system với EAR thresholds
- **Tham số**: `config_path` - File config, `eye_tracker` - EyeTracker instance
- **GUI Usage**: Gọi một lần với EyeTracker instance

### Update Methods
```python
def update() -> dict[str, Any]
```
- **Chức năng**: Cập nhật blink detection và phân tích trạng thái hiện tại
- **Trả về**: Dictionary với các keys:
  - `blink_detected`: Boolean detect được blink
  - `total_blinks`: Tổng số blink trong session
  - `blink_rate_per_minute`: Tần suất blink (blink/phút)
  - `avg_blink_duration`: Thời lượng blink trung bình
- **GUI Usage**: Gọi thường xuyên trong main loop để check blink events

### Statistics Methods
```python
def get_statistics() -> dict[str, Any]
```
- **Chức năng**: Lấy thống kê chi tiết về blink
- **GUI Usage**: Gọi để hiển thị comprehensive blink statistics

```python
def reset_statistics() -> None
```
- **Chức năng**: Reset tất cả thống kê cho session mới
- **GUI Usage**: Gọi khi bắt đầu monitoring session mới

---

# Module 4: Drowsiness Detector (`vision/drowsiness_detector.py`)

## Class: `DrowsinessDetector`
**Mục đích**: Phát hiện buồn ngủ sử dụng multi-signal analysis

### Initialization Methods
```python
def __init__(self, config_path: str = "settings.json")
```
- **Chức năng**: Khởi tạo multi-signal drowsiness detection với timers
- **GUI Usage**: Gọi một lần khi khởi tạo ứng dụng

### Update Methods
```python
def update(self, ear: Optional[float] = None, posture_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]
```
- **Chức năng**: Cập nhật drowsiness detection với data mới nhất
- **Tham số**: `ear` - Eye Aspect Ratio, `posture_data` - Kết quả posture analysis
- **Trả về**: Dictionary với các keys:
  - `drowsiness_detected`: Boolean detect được buồn ngủ
  - `reason`: Lý do detection ('ear_low', 'posture_bad', 'gaze_off')
  - `ear_duration`: Thời gian EAR thấp (giây)
  - `posture_bad_duration`: Thời gian posture kém (giây)
- **GUI Usage**: Gọi trong main loop để check drowsiness

### State Methods
```python
def is_drowsy() -> bool
```
- **Chức năng**: Kiểm tra trạng thái drowsiness hiện tại
- **GUI Usage**: Gọi để quick check trạng thái buồn ngủ

```python
def reset() -> None
```
- **Chức năng**: Reset tất cả internal state và timers
- **GUI Usage**: Gọi khi bắt đầu session mới

---

# Module 5: Health Data Collector (`vision/health_data_collector.py`)

## Class: `HealthDataCollector`
**Mục đích**: Thu thập và lưu trữ dữ liệu sức khỏe tự động vào CSV files

### Initialization Methods
```python
def __init__(self, collect_interval: float = 1.0, config_path: str = "settings.json", executor: Optional[ExecutorService] = None)
```
- **Chức năng**: Khởi tạo automated health data collection với thread-safe storage
- **Tham số**: `collect_interval` - Interval thu thập data (giây)
- **GUI Usage**: Gọi một lần để enable automatic data logging

### Control Methods
```python
def start_collection() -> None
```
- **Chức năng**: Bắt đầu automated data collection với background thread
- **GUI Usage**: Gọi khi bắt đầu monitoring session

```python
def stop_collection() -> None
```
- **Chức năng**: Dừng data collection và lưu session summary
- **GUI Usage**: Gọi khi kết thúc monitoring session

### Data Update Methods
```python
def update_health_data(self, health_data: Dict[str, Any]) -> None
```
- **Chức năng**: Cập nhật health data cho automatic logging (optimized 9 fields)
- **Tham số**: `health_data` - Dictionary với health metrics:
  - `timestamp`, `avg_ear`, `distance_cm`
  - `shoulder_tilt`, `head_pitch`, `head_yaw`
  - `drowsiness_detected`, `posture_status`
- **GUI Usage**: Gọi với consolidated health data từ tất cả modules

### Statistics Methods
```python
def get_current_stats() -> Dict[str, Any]
```
- **Chức năng**: Lấy thống kê session hiện tại
- **GUI Usage**: Gọi để hiển thị current session statistics

---

# Module 6: Main Application (`vision_app.py`)

## Class: `AEyeProVisionApp`
**Mục đích**: Main application class tích hợp tất cả vision modules

### Initialization Methods
```python
def __init__(self, config_file: str = "settings.json", show_camera: bool = True)
```
- **Chức năng**: Khởi tạo complete vision system với tất cả modules
- **Tham số**: `config_file` - Configuration file path, `show_camera` - Enable camera display
- **GUI Usage**: Gọi một lần để tạo main application instance

### Module Management Methods
```python
def initialize_modules() -> bool
```
- **Chức năng**: Khởi tạo tất cả vision modules (EyeTracker, PostureAnalyzer, etc.)
- **GUI Usage**: Gọi trước khi bắt đầu main application loop

### Processing Methods
```python
def process_frame() -> Dict[str, Any]
```
- **Chức năng**: Xử lý một camera frame qua tất cả modules
- **Trả về**: Dictionary với kết quả từ tất cả modules:
  - `eye_data`: Dữ liệu từ EyeTracker
  - `posture_data`: Dữ liệu từ PostureAnalyzer
  - `blink_data`: Dữ liệu từ BlinkDetector
  - `drowsy_data`: Dữ liệu từ DrowsinessDetector
- **GUI Usage**: Gọi trong main loop để frame processing

```python
def display_camera_feed(self, frame_result: Dict[str, Any])
```
- **Chức năng**: Hiển thị camera feed với comprehensive UI overlay
- **GUI Usage**: Gọi để hiển thị main monitoring interface

### Session Management Methods
```python
def setup_session_logging(self)
```
- **Chức năng**: Thiết lập session logging và data storage
- **Trả về**: Session ID string
- **GUI Usage**: Gọi khi bắt đầu mới session

```python
def save_session_summary(self)
```
- **Chức năng**: Lưu session summary vào CSV file
- **GUI Usage**: Gọi khi kết thúc session để lưu results

### Statistics Methods
```python
def update_statistics(self, frame_result: Dict[str, Any])
```
- **Chức năng**: Cập nhật thống kê từ frame result
- **GUI Usage**: Gọi sau mỗi processed frame

### Control Methods
```python
def shutdown(self)
```
- **Chức năng**: Gracefully shutdown tất cả modules
- **GUI Usage**: Gọi khi closing ứng dụng

---

# Module 7: Utilities (`utils/utils.py`)

## Configuration Functions
```python
def get_config(config_file='settings.json') -> Dict
```
- **Chức năng**: Load configuration từ JSON file
- **GUI Usage**: Gọi để load application settings

## Data Storage Functions
```python
def append_csv_row(row_dict, file_path, fieldnames=None)
```
- **Chức năng**: Append data row vào CSV file với thread safety
- **GUI Usage**: Gọi cho custom data logging

```python
def save_data(data, file_path)
```
- **Chức năng**: Save data vào JSON file với NumPy conversion
- **GUI Usage**: Gọi để save complex data structures

## Thread Management
```python
class ExecutorService
```
- **Chức năng**: Thread pool cho concurrent operations
- **GUI Usage**: Sử dụng cho background processing tasks

```python
def submit(self, fn, *args, **kwargs)
```
- **Chức năng**: Submit function để thực hiện trong thread pool
- **GUI Usage**: Sử dụng cho non-blocking operations

---

# 🎯 GUI Integration Workflow

## Basic Integration Steps:
1. **Initialize**: `app = AEyeProVisionApp(show_camera=False)`
2. **Setup Modules**: `app.initialize_modules()`
3. **Start Session**: `app.setup_session_logging()`
4. **Main Loop**:
   - Process: `frame_result = app.process_frame()`
   - Update Stats: `app.update_statistics(frame_result)`
   - Save Data: `app.save_frame_data(frame_result, 0)`
5. **Shutdown**: `app.shutdown()`

## Individual Module Usage:
1. **Eye Tracking**: `eye_tracker = EyeTracker()`
2. **Posture Analysis**: `posture_analyzer = PostureAnalyzer()`
3. **Blink Detection**: `blink_detector = BlinkDetector(eye_tracker)`
4. **Data Collection**: `health_collector = HealthDataCollector()`

---

# 📞 Troubleshooting

## Common Issues:
1. **Camera issues**: Kiểm tra `camera_index` trong config (try 0, 1, 2, 3)
2. **Import errors**: Đảm bảo Python 3.8+ và virtual environment được activate
3. **Performance issues**: Giảm frame_rate trong config hoặc sử dụng GPU acceleration
4. **Memory issues**: Gọi `shutdown()` properly on exit

## Getting Help:
- Kiểm tra console output cho error messages
- Xem logs trong `data/` directory
- Ensure all dependencies được cài đặt đúng version
- Test với `python vision_app.py` trước khi custom GUI

---

**AEyePro Version**: 3.0.0
**Python Requirements**: 3.8+
**License**: Proprietary