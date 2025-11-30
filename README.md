# 🤖 Báo cáo Bài tập nhóm Môn Trí tuệ Nhân tạo

**📋 Thông tin:**

* **📚 Môn học:** MAT3518 - Nhập môn Trí tuệ Nhân tạo  
* **📅 Học kỳ:** Học kỳ 1 - 2025-2026 
* **🏫 Trường:** VNU-HUS (Đại học Quốc gia Hà Nội - Trường Đại học Khoa học Tự nhiên)  
* **📝 Tiêu đề:** Ứng dụng theo dõi và tư vấn sức khỏe kỹ thuật số  
* **📅 Ngày nộp:** 30/11/2025 
* **📄 Báo cáo PDF:** 📄 [Báo cáo PDF](https://github.com/tmanhococ/Final_AI_Project/tree/master/Report)  
* **🖥️ Slide thuyết trình:** 🖥️ [Slide thuyết trình](https://github.com/tmanhococ/Final_AI_Project/tree/master/Slide)
* **📂 Kho lưu trữ:** 📂 https://github.com/tmanhococ/Final_AI_Project

**👥 Thành viên nhóm:**

| 👤 Họ và tên      | 🆔 Mã sinh viên     | 🐙 Tên GitHub        | 🛠️ Đóng góp  |
|------------------|--------------------|----------------------|----------------------|
| Lê Tiến Mạnh      | 23001535          | tmanhococ           | Leader, Chatbot Developer        |
| Trần Minh Đức      | 23001518          | tranminhduc9           | Frontend Dev, Deploy Server        |
| Hoàng Văn Phú      | 23001546          | phuhoangg           | Computer Vision Developer       |

---

## 📑 Tổng quan cấu trúc báo cáo

---

### Chương 1: Giới thiệu

#### 📝 Tóm tắt dự án
- **Tổng quan:** Dự án AEye Pro là hệ thống theo dõi và nâng cao sức khỏe người dùng khi sử dụng máy tính, tập trung vào các vấn đề như mỏi mắt và sai tư thế.
- **Mục tiêu:** Ghi nhận chỉ số sinh lý/hành vi thời gian thực và cung cấp cảnh báo kịp thời, đồng thời xây dựng hệ thống trợ lý ảo giúp đưa ra các lời khuyên y tế kịp thời cho người dùng.
- **Kết quả:** Xây dựng thành công ứng dụng web tích hợp Computer Vision để giám sát và Chatbot RAG để tư vấn cá nhân hóa.

#### ❓ Bài toán đặt ra
- **Vấn đề:** Làm việc lâu với máy tính gây mỏi mắt, đau cổ vai gáy, nhưng các giải pháp hiện tại thiếu khả năng phân tích dữ liệu lịch sử và gợi ý cá nhân hóa.
- **Ý nghĩa:** Hệ thống giúp phát hiện sớm rủi ro sức khỏe, điều chỉnh thói quen làm việc và giảm nguy cơ rối loạn cơ xương khớp, đưa ra lời khuyên y tế hữu ích cho người dùng.

---

### Chương 2: Phương pháp

#### ⚙️ Phương pháp
- **Cách tiếp cận:** Chia thành 2 bài toán cốt lõi: Giám sát hành vi thời gian thực (Computer Vision) và Trợ lý ảo tư vấn sức khỏe (Adaptive RAG).
- **Thuật toán thị giác:** Sử dụng MediaPipe (cân bằng hiệu năng/độ chính xác) để lấy điểm landmarks và OpenCV để xử lý ảnh nền tảng.  
  Các chỉ số đánh giá gồm: EAR (mắt), độ lệch vai, góc gập/nghiêng đầu và một vài thông số suy ra từ EAR.
- **Kiến trúc Chatbot:** Sử dụng Adaptive RAG với cơ chế định tuyến động (Routing) và tự sửa lỗi (Self-correction) thay vì RAG tuyến tính truyền thống.

---

### Chương 3: Triển khai

#### 💻 Triển khai
- **Kiến trúc hệ thống:** Mô hình Client-Server tách biệt.  
  Frontend (HTML/JS) hiển thị, Backend (Python/Flask) xử lý logic.
- **Giao thức giao tiếp:** Sử dụng song song RESTful API (quản trị, cài đặt) và Socket.IO (truyền luồng video và chỉ số thời gian thực) để giảm độ trễ.
- **Công cụ:** LangGraph quản lý luồng hội thoại có trạng thái, ChromaDB lưu trữ vector, và Pandas Agent để xử lý dữ liệu dạng bảng (CSV).

---

### Chương 4: Kết quả & Phân tích

#### 📊 Kết quả & Thảo luận
- **Thị giác máy tính:** Hoạt động ổn định, xử lý thời gian thực tốt.  
  Nhận diện chính xác các trạng thái: khoảng cách quá gần/xa, tư thế ngồi sai (nghiêng/cúi đầu), và dấu hiệu buồn ngủ (EAR thấp).
- **Chatbot:** Hoạt động hiệu quả theo 3 luồng:  
  - Giao tiếp (phản hồi nhanh).
  - Truy xuất kiến thức y khoa (chính xác từ tài liệu).
  - Phân tích dữ liệu cá nhân (tính toán thống kê từ log người dùng).

---

### Chương 5: Kết luận

#### ✅ Kết luận & Hướng phát triển
- **Tổng kết:** Đã hoàn thiện hệ thống đa luồng kết hợp AI tạo sinh và thị giác máy tính, giải quyết tốt bài toán tích hợp dữ liệu đa phương thức và có tính ứng dụng cao trong thực tế.
- **Hướng phát triển:**  
  - Ngắn hạn sẽ tối ưu UX (dashboard) và đóng gói bộ cài đặt.  
  - Dài hạn hướng tới phát triển mobile, đồng bộ cloud và tích hợp mô hình ngôn ngữ nhỏ (SLM) để chạy offline giúp tăng khả năng bảo mật hệ thống.


### Tài liệu tham khảo & Phụ lục
   ## 📚 Tài liệu tham khảo
- OpenCV Documentation: https://docs.opencv.org/
- MediaPipe Documentation: https://developers.google.com/mediapipe
- LangChain Documentation: https://python.langchain.com/
- ChromaDB Documentation: https://docs.trychroma.com/
- Gemini API Documentation: https://ai.google.dev/
- Các bài báo khoa học về việc nghiên cứu các mô hình thị giác máy tính cho tác vụ giám sát mắt và tư thế.
- Các bài báo nghiên cứu về RAG (Retrieval-Augmented Generation) trong ứng dụng chatbot y tế.

---




