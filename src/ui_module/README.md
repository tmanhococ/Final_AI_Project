# UI Module - AeyePro

Giao diện UI cho hệ thống giám sát sức khỏe bằng camera, hiển thị real-time và điều khiển cài đặt.

##  Cấu trúc

- **`index.html`**: Giao diện chính với sidebar metrics, camera, phần cài đặt, chatbot
- **`script.js`** : WebSocket để xử lý dữ liệu thời gian thực, API gọi API, điều khiển UI, hiện thông báo
- **`styles.css`**: Dùng để chỉnh theme cho ứng dụng

## Tính năng chính

- **7 Health Metrics**: Blink Rate, EAR, Distance, Shoulder/Head angles, Drowsy events
- **6 Settings**: Frame Rate, Camera Index, EAR Thresholds, Min/Max Distance
- **Camera Controls**: On/Off camera, Toggle landmarks (face mesh)
- **Real-time**: WebSocket streaming (~15 FPS video, 2 Hz metrics)
- **Notifications**: Popup alerts
- **Chatbot**: Chatbot module

## API Endpoints

**REST:**
- `POST /api/camera/start|stop` - Bật/tắt camera
- `GET /api/settings` - Lấy cài đặt
- `POST /api/settings` - Lưu cài đặt
- `POST /api/settings/face-mesh` - Bật tắt landmarks
- `POST /api/chatbot/message` - Gửi tin nhắn đến chatbot

**WebSocket:**
- `camera_frame` - Video frames
- `health_metrics` - Dữ liệu sức khỏe
- `system_status` - Trạng thái hệ thống

##  Sử dụng

Truy cập: `http://localhost:5000/`
