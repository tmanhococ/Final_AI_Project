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
    "health_monitoring": { ... },
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

**Hành vi:**
1. Validate cấu trúc JSON
2. Backup settings hiện tại
3. Ghi settings mới vào file
4. Nếu `reload_vision: true` và hệ thống đang chạy:
   - Dừng hệ thống
   - Khởi động lại với settings mới

---

#### **POST** `/api/settings/face-mesh`
Bật/tắt hiển thị landmarks trên khuôn mặt.

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
Stream các chỉ số sức khỏe.

**Dữ liệu:**
```json
{
  "eye_aspect_ratio": {
    "left": 0.25,
    "right": 0.26,
    "average": 0.255
  },
  "blink_count": 15,
  "blink_rate": 18.5,
  "posture": {
    "head_tilt": 5.2,
    "head_rotation": -2.3,
    "status": "good"
  },
  "drowsiness": {
    "level": 0.2,
    "alert": false
  },
  "timestamp": 1638000000.123
}
```

**Tần suất:** Mỗi 0.5 giây

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
2. Server: Lưu settings mới
3. Server: Dừng hệ thống hiện tại
4. Server: Khởi động lại với cấu hình mới
```
---

## Cách sử dụng

```python
python backend_server.py 
```

Sau đó truy cập localhost://127.0.0.1:5000/
