# AEyePro Configuration Documentation

## Overview
File `settings.json` chứa tất cả các tham số cấu hình cho hệ thống AEyePro, được chia thành 2 phần chính:
- `health_monitoring`: Các tham số cho computer vision và health monitoring
- `ui_settings`: Các tham số cho giao diện người dùng

---

## Health Monitoring Configuration

### Camera & Detection Settings
- **min_detection_confidence**: 0.8
  - Mức độ tin cậy tối thiểu để phát hiện khuôn mặt/pose (0.0-1.0)
  - Giá trị cao hơn = phát hiện chính xác hơn nhưng có thể bỏ sót
  - MediaPipe đề xuất: 0.7-0.9

- **min_tracking_confidence**: 0.8
  - Mức độ tin cậy tối thiểu để tracking liên tục (0.0-1.0)
  - Giá trị cao hơn = tracking ổn định hơn nhưng có thể lag
  - MediaPipe đề xuất: 0.7-0.9

- **pose_detection_confidence**: 0.8
  - Mức độ tin cậy tối thiểu để phát hiện pose cơ thể (0.0-1.0)
  - Ảnh hưởng đến độ chính xác của posture analysis

- **pose_tracking_confidence**: 0.8
  - Mức độ tin cậy tối thiểu để tracking pose liên tục (0.0-1.0)
  - Ảnh hưởng đến độ mượt của posture tracking

- **frame_rate**: 30
  - Tốc độ xử lý frame mỗi giây (FPS)
  - Cao hơn = mượt hơn nhưng tốn nhiều CPU hơn
  - Khuyến nghị: 25-30 cho laptop, 15-25 cho CPU yếu

- **camera_index**: 0
  - Index của camera device
  - 0 = camera mặc định, 1 = camera thứ hai, v.v.
  - Cần thay đổi nếu có nhiều camera hoặc camera không hoạt động

### Eye Tracking Configuration
- **LEFT_EYE**: [33, 160, 158, 133, 144, 153]
  - 6 landmarks MediaPipe cho mắt trái theo thứ tự:
    1, 2: Góc trên trong cùng
    3, 4: Góc dưới cùng
    5, 6: Góc ngoài cùng
  - Dựa trên MediaPipe Face Mesh 468 landmarks

- **RIGHT_EYE**: [362, 385, 387, 263, 373, 380]
  - 6 landmarks MediaPipe cho mắt phải theo thứ tự tương tự LEFT_EYE
  - Sử dụng cùng thứ tự landmarks như mắt trái

- **BLINK_THRESHOLD**: 0.27
  - Ngưỡng Eye Aspect Ratio (EAR) để phát hiện chớp mắt
  - EAR < threshold = mắt đang nhắm (chớp)
  - Tự động điều chỉnh qua calibration

- **DROWSY_THRESHOLD**: 0.30
  - Ngưỡng EAR để phát hiện buồn ngủ
  - EAR < threshold trong thời gian dài = buồn ngủ
  - Tự động điều chỉnh qua calibration

- **consecutive_frames**: 2
  - Số frames liên tục EAR < BLINK_THRESHOLD để xác nhận là chớp mắt
  - Tăng lên để giảm false positive, giảm xuống để tăng sensitivity

- **max_blink_duration**: 0.5
  - Thời gian tối đa cho một cái chớp (giây)
  - Blink kéo dài hơn sẽ bị loại bỏ

- **min_blink_interval**: 0.1
  - Khoảng thời gian tối thiểu giữa các chớp mắt (giây)
  - Ngăn phát hiện nhiều chớp trong cùng khoảng thời gian

### Distance & Calibration
- **camera_focal_length**: 600
  - Tiêu cự camera (pixels) cho ước tính khoảng cách
  - Tính từ: focal_length = (known_distance * known_pixel_size) / known_object_size
  - Calibration tự động sẽ điều chỉnh giá trị này

- **AVERAGE_EYE_DISTANCE_CM**: 6.3
  - Khoảng cách trung bình giữa hai mắt (cm)
  - Dùng cho ước tính khoảng cách đến camera
  - Giá trị trung bình cho người trưởng thành: 6.0-6.5 cm

- **MIN_EYE_PIXEL_DISTANCE**: 30
  - Khoảng cách tối thiểu giữa hai mắt trên ảnh (pixels)
  - Dùng để lọc ra các phát hiện không hợp lệ

- **MIN_REASONABLE_DISTANCE**: 50
  - Khoảng cách tối thiểu hợp lý đến màn hình (cm)
  - Khoảng cách quá gần sẽ được cảnh báo

- **MAX_REASONABLE_DISTANCE**: 80
  - Khoảng cách tối đa hợp lý đến màn hình (cm)
  - Khoảng cách quá xa sẽ được cảnh báo

- **EPSILON**: 1e-7
  - Giá trị rất nhỏ để tránh chia cho 0 trong tính toán
  - Không cần thay đổi giá trị này

### Posture Analysis Configuration
- **max_head_updown_angle**: 22
  - Góc nghiêng đầu lên/xuống tối đa (độ)
  - Vượt quá sẽ được cảnh báo tư thế kém
  - 0 = thẳng, dương = nghiêng xuống, âm = ngửa lên

- **max_head_side_angle**: 20
  - Góc nghiêng đầu trái/phải tối đa (độ)
  - Vượt quá sẽ được cảnh báo tư thế kém
  - 0 = thẳng, dương = nghiêng phải, âm = nghiêng trái

- **max_shoulder_tilt**: 15
  - Góc nghiêng vai tối đa (độ)
  - Vượt quá sẽ được cảnh báo tư thế kém
  - 0 = cân bằng, dương = nghiêng phải, âm = nghiêng trái

### Session & Data Management
- **session_duration_threshold**: 3600
  - Thời lượng tối đa cho một session (giây)
  - 3600 = 1 giờ, session dài hơn sẽ tự động lưu và bắt đầu mới

- **data_retention_days**: 7
  - Số ngày giữ lại dữ liệu cũ
  - Tự động xóa files cũ hơn số ngày này

- **drowsy_ear_duration**: 1.5
  - Thời gian EAR thấp để kích hoạt cảnh báo buồn ngủ (giây)
  - EAR < DROWSY_THRESHOLD trong thời gian này = buồn ngủ

---

## UI Settings Configuration

### Appearance & Behavior
- **theme**: "dark"
  - Giao diện màu sắc ("dark", "light", "auto")
  - "auto" sẽ theo hệ điều hành

- **language**: "vi"
  - Ngôn ngữ giao diện ("vi", "en")
  - "vi" = Tiếng Việt, "en" = English

- **enable_notifications**: true
  - Bật/tắt thông báo hệ thống
  - Bao gồm cảnh báo tư thế, buồn ngủ, v.v.

- **auto_refresh_interval**: 30
  - Khoảng thời gian tự động refresh giao diện (giây)
  - Giá trị nhỏ hơn = cập nhật thường xuyên hơn nhưng tốn tài nguyên hơn

---

## Calibration Guide

### Automatic Calibration
1. Chạy calibration trong AEyePro (nhấn 'c')
2. Nhìn thẳng vào camera trong 10 giây
3. Chớp mắt bình thường
4. Hệ thống sẽ tự động điều chỉnh:
   - BLINK_THRESHOLD
   - DROWSY_THRESHOLD
   - camera_focal_length

### Manual Calibration
- **BLINK_THRESHOLD**: Bắt đầu từ 0.25, điều chỉnh +/- 0.02
- **DROWSY_THRESHOLD**: Bắt đầu từ 0.30, điều chỉnh +/- 0.03
- **camera_focal_length**: Đo từ khoảng cách known, tính lại focal_length

### Testing Calibration
- Chớp mắt và kiểm tra detection
- Nhìn xa/gần để test distance estimation
- Nghiêng đầu để test posture detection

---

## Performance Optimization

### For Low-End Systems
```json
{
  "frame_rate": 15,
  "min_detection_confidence": 0.7,
  "min_tracking_confidence": 0.7
}
```

### For High-End Systems
```json
{
  "frame_rate": 60,
  "min_detection_confidence": 0.9,
  "min_tracking_confidence": 0.9
}
```

### For Different Camera Types
- **Webcam HD (1920x1080)**: focal_length = 800-1000
- **Webcam Standard (640x480)**: focal_length = 500-600
- **Laptop Camera**: focal_length = 400-500

---

## Troubleshooting

### Camera Issues
- **Problem**: Không phát hiện khuôn mặt
  **Solution**: Giảm `min_detection_confidence` xuống 0.6, thử `camera_index` khác

- **Problem**: Tracking không ổn định
  **Solution**: Giảm `min_tracking_confidence` xuống 0.6, giảm `frame_rate`

### Detection Issues
- **Problem**: Nhiều false positive chớp mắt
  **Solution**: Tăng `consecutive_frames` lên 3, tăng `BLINK_THRESHOLD`

- **Problem**: Bỏ sót chớp mắt
  **Solution**: Giảm `consecutive_frames` xuống 1, giảm `BLINK_THRESHOLD`

### Performance Issues
- **Problem**: CPU usage cao
  **Solution**: Giảm `frame_rate` xuống 15-20, tăng confidence thresholds

- **Problem**: Lag khi xử lý
  **Solution**: Giảm `frame_rate`, tắt các features không cần thiết

---

## Environment Variables Override

Có thể override các tham số bằng environment variables:

```bash
export AEYE_CAMERA_INDEX=1
export AEYE_DATA_RETENTION_DAYS=30
export AEYE_FORCE_CPU=true
```

Ưu tiên: Environment Variables > settings.json > default values

---

## Version Information
- **File Version**: 3.0.0
- **Compatible AEyePro Version**: 3.0.0+
- **Last Updated**: 2025

---

**Note**: Thay đổi các tham số này có thể ảnh hưởng đến độ chính xác và hiệu suất của hệ thống. Khuyến nghị chạy calibration sau khi thay đổi các tham số quan trọng.