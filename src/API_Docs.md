# AEyePro API Documentation

Tài liệu API cho hệ thống AEyePro Backend Server.

---

## Base URL
```
http://localhost:5000
```

---

## REST API Endpoints

### 1. Điều khiển Camera

#### **POST** `/api/camera/start`
Khởi động camera và hệ thống xử lý vision.

**Response thành công (200):**
```json
{
  "success": true,
  "message": "Vision system started",
  "session_id": "abc123"
}
```

**Response lỗi (400/500):**
```json
{
  "success": false,
  "message": "Error description",
  "error": "ERROR_CODE"
}
```

**Mã lỗi:**
- `ALREADY_RUNNING` - Hệ thống đã chạy
- `INITIALIZATION_FAILED` - Khởi tạo thất bại
- `CAMERA_UNAVAILABLE` - Camera không khả dụng

---

#### **POST** `/api/camera/stop`
Dừng camera và hệ thống xử lý vision.

**Response thành công (200):**
```json
{
  "success": true,
  "message": "Vision system stopped"
}
```

**Response lỗi (400):**
```json
{
  "success": false,
  "message": "Error description",
  "error": "NOT_RUNNING"
}
```

---

#### **GET** `/api/camera/status`
Lấy trạng thái hiện tại của hệ thống.

**Response (200):**
```json
{
  "is_running": true,
  "session_id": "abc123",
  "fps": 30.5,
  "last_error": null
}
```

---

### 2. Quản lý Cài đặt

#### **POST** `/api/settings/face-mesh`
Bật/tắt hiển thị face mesh landmarks trên khuôn mặt.

**Request:**
```json
{
  "enabled": true
}
```

**Response (200):**
```json
{
  "success": true,
  "face_mesh_enabled": true
}
```

**Mô tả:** 
- Endpoint này cho phép bật/tắt việc hiển thị các landmarks của khuôn mặt trên camera feed
- Thay đổi áp dụng ngay lập tức mà không cần reload vision system

---

#### **GET** `/api/settings`
Đọc cấu hình từ `config/settings.json`.

**Response thành công (200):**
```json
{
  "success": true,
  "settings": {
    "health_monitoring": { ... },
    "ui_settings": { ... }
  }
}
```

**Response lỗi (404/400/500):**
```json
{
  "success": false,
  "message": "Error description",
  "error": "ERROR_CODE"
}
```

**Mã lỗi:**
- `FILE_NOT_FOUND` - File settings.json không tồn tại
- `PARSE_ERROR` - Lỗi phân tích JSON

---

#### **POST** `/api/settings`
Cập nhật cấu hình và tải lại hệ thống.

**Request:**
```json
{
  "settings": {
    "health_monitoring": {
      "frame_rate": 30,
      "camera_index": 0,
      "DROWSY_THRESHOLD": 0.30,
      "BLINK_THRESHOLD": 0.27,
      "MIN_REASONABLE_DISTANCE": 20,
      "MAX_REASONABLE_DISTANCE": 150
    },
    "ui_settings": { ... }
  },
  "reload_vision": true
}
```

**Response thành công (200):**
```json
{
  "success": true,
  "message": "Settings updated successfully",
  "reloaded": true
}
```

**Các thông số cài đặt:**
- `frame_rate`: Tốc độ xử lý khung hình (10-60 FPS)
- `camera_index`: Chỉ số camera (0=mặc định, 1=camera ngoài)
- `DROWSY_THRESHOLD`: Ngưỡng EAR phát hiện buồn ngủ (0.20-0.40)
- `BLINK_THRESHOLD`: Ngưỡng EAR phát hiện chớp mắt (0.20-0.35)
- `MIN_REASONABLE_DISTANCE`: Khoảng cách tối thiểu an toàn (15-50 cm)
- `MAX_REASONABLE_DISTANCE`: Khoảng cách tối đa hiệu quả (80-200 cm)

**Hành vi:**
1. Validate cấu trúc JSON
2. Backup settings hiện tại vào `.json.backup`
3. Ghi settings mới vào file
4. Nếu `reload_vision: true` và hệ thống đang chạy:
   - Dừng hệ thống hiện tại
   - Khởi động lại với settings mới
5. Cập nhật global thresholds trong UI để áp dụng ngay lập tức

**Lưu ý:** 
- Các thay đổi về thresholds được đồng bộ với frontend để cảnh báo real-time
- Frontend sẽ tự động cập nhật các biến: `window.drowsyThreshold`, `window.blinkThreshold`, `window.minDistanceThreshold`, `window.maxDistanceThreshold`

---

## WebSocket Events

### Client → Server

#### `connect`
Kết nối WebSocket thành công.

**Response từ server:**
```json
{
  "is_running": true,
  "session_id": "abc123",
  "fps": 30.5
}
```

---

#### `request_status`
Yêu cầu trạng thái hệ thống.

**Emit:** `request_status`

**Response:** Event `system_status` với dữ liệu trạng thái.

---

### Server → Client

#### `camera_frame`
Stream khung hình camera (15 FPS).

**Dữ liệu:**
```json
{
  "frame": "base64_encoded_jpeg",
  "timestamp": 1638000000.123
}
```

**Tần suất:** ~15 FPS (mỗi 66ms)

---

#### `health_metrics`
Stream các chỉ số sức khỏe và cảnh báo real-time.

**Dữ liệu:**
```json
{
  "eye": {
    "avg_ear": 0.255,
    "distance_cm": 45.2
  },
  "blink": {
    "blink_count": 15,
    "blink_rate": 18.5
  },
  "posture": {
    "head_side_angle": 5.2,
    "head_updown_angle": -2.3,
    "shoulder_tilt": 1.5,
    "eye_distance_cm": 45.2,
    "status": "good"
  },
  "drowsiness": {
    "detected": false
  },
  "system": {
    "timestamp": 1638000000.123
  }
}
```

**Tần suất:** Mỗi 0.5 giây (2 Hz)

**Frontend Processing:**
- So sánh `distance_cm` với `window.minDistanceThreshold` → Cảnh báo **TOO CLOSE**
- So sánh `distance_cm` với `window.maxDistanceThreshold` → Cảnh báo **TOO FAR**
- Kiểm tra `drowsiness.detected` → Cảnh báo **DROWSINESS DETECTED**
- Kiểm tra `posture.status === "poor"` → Cảnh báo **BAD POSTURE**
- Tất cả cảnh báo có cooldown 10 giây để tránh spam

---

#### `system_status`
Cập nhật trạng thái hệ thống.

**Dữ liệu:**
```json
{
  "is_running": true,
  "session_id": "abc123",
  "fps": 30.5,
  "last_error": null
}
```

---

## Luồng hoạt động chính

### Khởi động hệ thống:
```
1. Client: POST /api/camera/start
2. Server: Khởi tạo camera + vision modules
3. Server: Bắt đầu broadcast frames (15 FPS) và metrics (2 Hz)
4. Client: Nhận event 'camera_frame' và 'health_metrics'
```

### Dừng hệ thống:
```
1. Client: POST /api/camera/stop
2. Server: Dừng camera + vision threads
3. Server: Ngừng broadcast
```

### Cập nhật cài đặt:
```
1. Client: POST /api/settings (với reload_vision: true)
2. Server: Lưu settings mới vào config/settings.json
3. Server: Dừng hệ thống hiện tại
4. Server: Khởi động lại với cấu hình mới
5. Client: Cập nhật global thresholds (drowsy, blink, minDist, maxDist)
6. Client: Áp dụng ngay cho cảnh báo real-time
```

### Luồng cảnh báo real-time:
```
1. Server: Broadcast health_metrics qua WebSocket (mỗi 0.5s)
2. Client: Nhận metrics và kiểm tra:
   - Distance < minDistanceThreshold → Popup "TOO CLOSE!"
   - Distance > maxDistanceThreshold → Popup "TOO FAR!"
   - drowsiness.detected → Popup "DROWSINESS DETECTED!"
   - posture.status === "poor" → Popup "BAD POSTURE!"
3. Client: Cooldown 10s giữa các notification cùng loại
```

---

---

## 3. Chatbot API

#### **POST** `/api/chatbot/message`
Gửi tin nhắn đến chatbot và nhận phản hồi.

**Request:**
```json
{
  "message": "What's my average blink rate?",
  "thread_id": "user_12345"
}
```

**Response thành công (200):**
```json
{
  "success": true,
  "response": "Your average blink rate is 18.5 blinks per minute...",
  "message": "Message processed successfully",
  "timestamp": 1638000000.123,
  "thread_id": "user_12345"
}
```

**Response lỗi:**
```json
{
  "success": false,
  "message": "Error description",
  "error": "ERROR_CODE"
}
```

**Mã lỗi:**
- `CHATBOT_UNAVAILABLE` - Module chatbot không được cài đặt
- `INVALID_REQUEST` - Thiếu trường bắt buộc hoặc JSON không hợp lệ
- `PROCESSING_ERROR` - Lỗi xử lý chatbot
- `SERVER_ERROR` - Lỗi server không mong đợi

---

#### **GET** `/api/chatbot/status`
Kiểm tra trạng thái chatbot.

**Response (200):**
```json
{
  "available": true,
  "initialized": true,
  "error": null
}
```

---

## Cách sử dụng

### Cài đặt thư viện:
```bash
pip install -r requirements.txt
```

### Khởi động server:
```bash
python backend_server.py
```

### Truy cập UI:
```
http://localhost:5000
```

### Cấu hình Settings:
1. Mở Settings Panel trong UI
2. Điều chỉnh các thông số:
   - **Frame Rate**: Tốc độ xử lý (↓ giảm CPU)
   - **Drowsy/Blink Threshold**: Độ nhạy phát hiện
   - **Min/Max Distance**: Ngưỡng cảnh báo khoảng cách
3. Nhấn **Save Settings** để áp dụng
4. Hệ thống tự động reload với cấu hình mới

### Xem Logs:
- Backend logs: Terminal chạy `backend_server.py`
- Frontend logs: Browser Console (F12)
- Notification logs: Tìm `[NOTIFICATION]` trong console
