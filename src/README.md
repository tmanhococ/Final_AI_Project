# AEyePro - Computer Vision Health Monitoring System

## Giới thiệu

Thư mục `src/` chứa source code chính của hệ thống **AEyePro** - một ứng dụng theo dõi sức khỏe mắt và tư thế sử dụng Computer Vision và AI Chatbot.

Hệ thống bao gồm:
- **Computer Vision Module**: Theo dõi mắt, tư thế, và phát hiện mệt mỏi thông qua webcam
- **AI Chatbot**: Trả lời câu hỏi về sức khỏe mắt và tư thế sử dụng RAG (Retrieval-Augmented Generation)
- **Web Interface**: Giao diện web real-time để hiển thị dữ liệu và tương tác với chatbot

---

## Cấu trúc thư mục

```
src/
├── main.py                 # Entry point - Flask server chạy tại localhost:5000
├── requirements.txt         # Dependencies cho vision module và web server
│
├── chatbot/                # 📁 AI Chatbot Module (RAG + LangGraph)
│   ├── app.py              # CLI application cho chatbot
│   ├── app_runtime.py      # Runtime assembly cho chatbot app
│   ├── chat_interface.py   # Public interface cho chatbot
│   ├── config.py           # Configuration management
│   ├── graph.py            # LangGraph workflow assembly
│   ├── llm_factory.py      # LLM và embeddings factory
│   ├── state.py            # GraphState definition
│   ├── nodes/              # LangGraph nodes
│   │   ├── chat_utils.py  # Guardrails, Social Bot, Contextualize
│   │   ├── query_analysis.py
│   │   ├── csv_node.py
│   │   ├── retriever_node.py
│   │   ├── grader_node.py
│   │   ├── generator_node.py
│   │   └── rewriter_node.py
│   ├── tools/              # Data loading utilities
│   │   ├── vector_store.py
│   │   └── csv_loader.py
│   ├── test/              # Test files
│   ├── requirements_chatbot.txt
│   └── .env               # Environment variables (cần tạo)
│
├── vision/                 # 📁 Computer Vision Module
│   ├── eye_tracker.py      # Eye tracking với MediaPipe
│   ├── posture_analyzer.py # Phân tích tư thế
│   ├── blink_detector.py   # Phát hiện chớp mắt
│   ├── drowsiness_detector.py # Phát hiện buồn ngủ
│   ├── health_data_collector.py # Thu thập dữ liệu sức khỏe
│   ├── vision_manager.py   # Vision manager (thread management)
│   └── vision_app.py       # Vision application
│
├── ui_module/              # 📁 Web Interface Module
│   ├── index.html          # HTML template
│   ├── script.js           # Frontend JavaScript
│   └── styles.css          # CSS styles
│
├── utils/                  # Utilities và helpers
│   ├── utils.py            # Common utilities
│   └── __init__.py
│
├── config/                 # Configuration files
│   ├── settings.json       # Application settings
│   └── settings_documentation.md
│
├── data/                   # Data storage
│   ├── medical_docs/       # Medical documents cho RAG
│   ├── chroma_db/          # ChromaDB vector store
│   ├── summary.csv         # Health logs summary
│   └── realtime_*.csv      # Real-time health logs
│
└── docs/                   # Documentation
    ├── Chatbot_Docs.md     # Chi tiết chatbot documentation
    ├── CV_Docs.md          # Computer Vision documentation
    └── UI_Docs.md          # UI documentation
```

### 3 Module chính

1. **`chatbot/`**: Module AI Chatbot sử dụng LangGraph và RAG để trả lời câu hỏi về sức khỏe
2. **`vision/`**: Module Computer Vision sử dụng MediaPipe để theo dõi mắt và tư thế
3. **`ui_module/`**: Module giao diện web (HTML/CSS/JS) để hiển thị dữ liệu real-time

---

## Hướng dẫn cài đặt và sử dụng

### Phần 1: Chạy bằng Virtual Environment (.venv)

#### Bước 1: Tạo file cấu hình `.env`

Tạo file `src/chatbot/.env` với nội dung sau:

```env
# Google Gemini API Key (BẮT BUỘC)
GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"

# LangSmith Tracing (Tùy chọn - để debug)
LANGCHAIN_TRACING_V2=false
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="YOUR_LANGSMITH_KEY_HERE"
LANGCHAIN_PROJECT="health-care-chatbot"

# Model Configuration (Tùy chọn - có thể override)
# LLM_MODEL_NAME="gemini-2.5-flash"
# EMBEDDING_MODEL_NAME="models/embedding-001"

# Data Paths (Tùy chọn - có thể override)
# CHROMA_PERSIST_DIRECTORY="./data/chroma_db"
# CSV_FILE_PATH="./data/logs/user_health_log.csv"
```

**Lưu ý**: Thay `YOUR_GOOGLE_API_KEY_HERE` bằng API key thật của bạn. Lấy API key tại: https://ai.google.dev/

#### Bước 2: Tạo và kích hoạt Virtual Environment

```powershell
# Di chuyển vào thư mục src
cd D:\AI_Final\Final_AI_Project\src

# Tạo virtual environment
python -m venv .venv

# Kích hoạt virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Windows CMD:
.\.venv\Scripts\activate.bat

# Linux/Mac:
source .venv/bin/activate
```

#### Bước 3: Cài đặt dependencies

```powershell
# Đảm bảo đã activate .venv
pip install --upgrade pip
pip install -r requirements.txt
```

#### Bước 4: Chạy ứng dụng

```powershell
# Đảm bảo đã activate .venv
python main.py
```

Ứng dụng sẽ chạy tại: **http://localhost:5000**

Mở trình duyệt và truy cập:
- **Web Interface**: http://localhost:5000
- **API Documentation**: http://localhost:5000/api/docs (nếu có)

---

## Yêu cầu hệ thống

- **Python**: 3.10+ (khuyến nghị 3.12)
- **Webcam**: Để sử dụng Computer Vision module
- **Google API Key**: Để sử dụng Chatbot module (miễn phí theo quota)
- **RAM**: Tối thiểu 4GB (khuyến nghị 8GB)
- **Disk Space**: ~2GB cho dependencies và data

---

## Troubleshooting

### Lỗi: `GOOGLE_API_KEY is not set`

**Giải pháp**: Tạo file `src/chatbot/.env` và thêm `GOOGLE_API_KEY="your_key_here"`

### Lỗi: `ModuleNotFoundError: No module named 'cv2'`

**Giải pháp**: 
- Đảm bảo đã activate `.venv` trong thư mục `src/`
- Chạy lại: `pip install -r requirements.txt`

### Lỗi: Port 5000 đã được sử dụng

**Giải pháp**: 
- Đóng ứng dụng khác đang dùng port 5000
- Hoặc thay đổi port trong `main.py`

---

## Tài liệu tham khảo

- **Chatbot Documentation**: Xem `docs/Chatbot_Docs.md` để biết chi tiết về chatbot module
- **Computer Vision Documentation**: Xem `docs/CV_Docs.md` để biết chi tiết về vision module
- **UI Documentation**: Xem `docs/UI_Docs.md` để biết chi tiết về web interface

---

## License

```
MIT License

Copyright (c) 2024 AEyePro Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Liên hệ và đóng góp

Nếu bạn gặp vấn đề hoặc muốn đóng góp, vui lòng tạo issue hoặc pull request trên repository.

**AEyePro Team** - 11/2025

Le Tien Manh

Tran Minh Duc

Hoang Van Phu

