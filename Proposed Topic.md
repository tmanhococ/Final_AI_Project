## 🇻🇳 Phiên bản Tiếng Việt

### 🏷️ Tên nhóm
Nhóm 12 - AEYEPRO

### 📝 Tên dự án
Ứng dụng theo dõi và tư vấn sức khỏe kỹ thuật số

### 👥 Thành viên nhóm
| 👤 Họ và tên 🧑‍🎓  | 🆔 Mã sinh viên 🧾 | 🐙 Tên GitHub 🔗     |
|------------------|---------------------|---------------------|
| Lê Tiến Mạnh      | 23001535    | tmanhococ      |
| Trần Minh Đức      | 23001518    | tranminhduc9      |
| Hoàng Văn Phú      | 23001546    | phuhoangg      |

### 🗒️ Tóm tắt
Hệ thống cung cấp cảnh báo kịp thời khi phát hiện dấu hiệu mệt mỏi, đồng thời lưu trữ dữ liệu để phân tích dài hạn. Chatbot tích hợp dựa trên kiến trúc Retrieval-Augmented Generation (RAG) có khả năng truy vấn lịch sử, đưa ra thống kê, nhận định và gợi ý cá nhân hóa dựa trên thói quen thực tế của người dùng. Về mặt kỹ thuật, AEYEPro ứng dụng OpenCV cho xử lý hình ảnh và MediaPipe cho việc lấy các điểm dữ liệu trên cơ thể người, LangChain và LangGraph kết hợp Vector Store ChromaDB cho RAG và Gemini API cho mô hình ngôn ngữ, triển khai dưới dạng ứng dụng desktop để đảm bảo hiệu suất, bảo mật và vận hành liên tục.

### 🎯 Bối cảnh
Trong thực tế, nhiều người gặp khó khăn trong việc duy trì tư thế đúng và kiểm soát dấu hiệu mệt mỏi mắt khi làm việc lâu với màn hình. Giải pháp hiện tại thường chỉ cung cấp cảnh báo đơn giản, thiếu khả năng phân tích dữ liệu lịch sử và gợi ý cá nhân hóa.

AEYEPro hướng tới việc khắc phục những hạn chế này bằng một hệ thống theo dõi thời gian thực. Hệ thống ghi nhận dữ liệu sinh lý và hành vi, phân tích tư thế và mức độ mệt mỏi, đồng thời cung cấp thông tin phản hồi và khuyến nghị cá nhân hóa dựa trên dữ liệu thu thập được. Cách tiếp cận này không chỉ giúp người dùng phát hiện sớm các vấn đề sức khỏe mà còn hỗ trợ điều chỉnh thói quen làm việc, nâng cao hiệu quả và giảm nguy cơ các rối loạn cơ xương khớp hoặc thị giác khi phải tiếp xúc liên tục với màn hình máy tính.

### 🚀 Kế hoạch
Dự án được thực hiện theo các bước chính sau:

1. **Đặt vấn đề và nghiên cứu**: Xác định các vấn đề sức khỏe khi làm việc lâu với màn hình, đặt câu hỏi nghiên cứu về phương pháp theo dõi và tư vấn hiệu quả.

2. **Tìm giải pháp**: Nghiên cứu các công nghệ phù hợp (OpenCV, MediaPipe, RAG, LLM) và xác định kiến trúc hệ thống tối ưu.

3. **Lên ý tưởng và thiết kế module**: Thiết kế các module chính bao gồm:
   - Module theo dõi thời gian thực (camera, xử lý ảnh)
   - Module phát hiện tư thế và mệt mỏi
   - Module lưu trữ và phân tích dữ liệu
   - Module chatbot RAG với khả năng tư vấn cá nhân hóa
   - Module giao diện người dùng

4. **Triển khai và thử nghiệm**: Xây dựng từng module, tích hợp các thành phần và tiến hành thử nghiệm ban đầu với người dùng thực tế. Sử dụng mô trường ảo (.venv) để tiến hành thử nghiệm và tối ưu.

5. **Tối ưu và kiểm thử**: Đánh giá hiệu suất, độ chính xác của hệ thống, tối ưu hóa thuật toán và giao diện người dùng, thực hiện kiểm thử toàn diện trước khi hoàn thiện.

### 📚 Tài liệu tham khảo
- OpenCV Documentation: https://docs.opencv.org/
- MediaPipe Documentation: https://developers.google.com/mediapipe
- LangChain Documentation: https://python.langchain.com/
- ChromaDB Documentation: https://docs.trychroma.com/
- Gemini API Documentation: https://ai.google.dev/
- Các bài báo khoa học về phát hiện mệt mỏi và tư thế cơ thể
- Nghiên cứu về RAG (Retrieval-Augmented Generation) trong ứng dụng chatbot y tế
