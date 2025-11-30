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

### Phần 2: Chạy bằng Docker ⚠️ (Đang thử nghiệm)

> **⚠️ CẢNH BÁO QUAN TRỌNG**: 
> - Docker setup hiện tại đang trong **giai đoạn thử nghiệm** và có thể vẫn còn lỗi
> - **Computer Vision module (vision/)** **KHÔNG được container hóa** vì:
>   - Truy cập webcam từ container phức tạp và không ổn định trên Windows/macOS
>   - MediaPipe và OpenCV yêu cầu nhiều system dependencies khó cấu hình trong container
>   - Device passthrough chỉ hoạt động ổn định trên Linux
> - **Chỉ Backend + Chatbot** được chạy trong Docker container
> - **Vision module phải được cài đặt và chạy thủ công trên host machine** (xem hướng dẫn bên dưới)

Docker cho phép bạn chạy **Backend API và Chatbot** trong môi trường container hóa, đảm bảo tính nhất quán giữa các môi trường khác nhau. Tuy nhiên, do hạn chế kỹ thuật, Computer Vision module vẫn cần chạy native trên host.

#### Yêu cầu

- **Docker Engine**: 20.10+ (cài đặt tại https://www.docker.com/get-started)
- **Docker Compose**: 2.0+ (thường đi kèm với Docker Desktop)
- **Disk Space**: ~3GB cho Docker image và dependencies
- **Python trên host**: Để chạy Vision module (nếu cần sử dụng webcam)

#### Bước 1: Tạo file cấu hình `.env`

Tạo file `src/chatbot/.env` với nội dung (giống như Phần 1):

```env
# Google Gemini API Key (BẮT BUỘC)
GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"

# LangSmith Tracing (Tùy chọn)
LANGCHAIN_TRACING_V2=false
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="YOUR_LANGSMITH_KEY_HERE"
LANGCHAIN_PROJECT="health-care-chatbot"
```

**Lưu ý**: Thay `YOUR_GOOGLE_API_KEY_HERE` bằng API key thật của bạn.

#### Bước 2: Cài đặt Vision Module trên Host (Nếu cần sử dụng webcam)

**QUAN TRỌNG**: Nếu bạn muốn sử dụng Computer Vision module (theo dõi mắt, tư thế), bạn **PHẢI** cài đặt các thư viện cho module này trên host machine (không phải trong container):

```powershell
# Từ thư mục src/
cd D:\AI_Final\Final_AI_Project\src

# Activate virtual environment (nếu chưa có thì tạo như Phần 1)
.\.venv\Scripts\Activate.ps1

# Cài đặt các thư viện cho Computer Vision
pip install opencv-python>=4.8.0
pip install mediapipe==0.10.14
pip install numpy>=1.21.0
pip install pandas>=1.3.0

# Hoặc cài tất cả từ requirements.txt
pip install -r requirements.txt
```

**Lý do**: Vision module cần truy cập trực tiếp vào webcam và các system libraries (OpenCV, MediaPipe) hoạt động tốt nhất khi chạy native trên host, không phải trong container.

#### Bước 3: Build và chạy Backend + Chatbot với Docker Compose

Từ **thư mục gốc của project** (nơi có `docker-compose.yml`):

```bash
# Build image và khởi động container (chỉ Backend + Chatbot)
docker-compose up -d --build

# Xem logs
docker-compose logs -f backend

# Dừng container
docker-compose down
```

> **Lưu ý**: Container này chỉ chạy Backend API và Chatbot. Vision module sẽ chạy riêng trên host (nếu cần).

#### Bước 4: Kiểm tra ứng dụng

Sau khi container chạy (khoảng 30-60 giây để khởi động):

- **Web Interface**: http://localhost:5000
- **Backend API**: http://localhost:5000/api/...
- **Health Check**: Container tự động kiểm tra sức khỏe mỗi 30 giây

**Lưu ý**: 
- Backend và Chatbot sẽ hoạt động bình thường
- **Vision module sẽ KHÔNG hoạt động** nếu chỉ chạy container (cần cài đặt và chạy riêng trên host)

**Hoặc chạy script test tự động** (từ thư mục project root):

```bash
# Linux/Mac:
chmod +x test_docker.sh
./test_docker.sh

# Windows PowerShell:
.\test_docker.ps1
```

Script sẽ tự động:
- Kiểm tra Docker và Docker Compose đã cài đặt
- Kiểm tra file `.env` và thư mục `data/`
- Build và khởi động container
- Kiểm tra health status
- Test HTTP endpoint

#### Các lệnh Docker hữu ích

```bash
# Xem logs real-time
docker-compose logs -f backend

# Xem trạng thái container
docker-compose ps

# Restart container
docker-compose restart backend

# Dừng và xóa container (giữ lại data volumes)
docker-compose down

# Dừng và xóa tất cả (bao gồm volumes - CẨN THẬN!)
docker-compose down -v

# Rebuild image từ đầu (không dùng cache)
docker-compose build --no-cache

# Vào trong container để debug
docker-compose exec backend bash
```

#### Cấu trúc Docker

```
Project Root/
├── Dockerfile              # Multi-stage build cho backend
├── docker-compose.yml      # Orchestration cho services
├── .dockerignore          # Files bỏ qua khi build
└── src/
    ├── chatbot/
    │   └── .env           # Environment variables (không commit)
    └── data/              # Mounted volume (persist ChromaDB)
```

#### Lưu trữ dữ liệu (Volumes)

Docker Compose tự động mount thư mục `src/data/` vào container để:
- **ChromaDB vector store** được lưu trữ bền vững
- **CSV logs** được giữ lại khi container restart
- **Medical documents** có sẵn cho RAG

Dữ liệu được lưu tại `./src/data/` trên host machine.

#### Chạy Vision Module (Trên Host - Không container hóa)

**⚠️ QUAN TRỌNG**: Vision module **KHÔNG được khuyến nghị chạy trong container** vì:
- Phức tạp và không ổn định khi truy cập webcam từ container
- Yêu cầu nhiều system dependencies khó cấu hình
- Chỉ hoạt động ổn định trên Linux với device passthrough

**Cách chạy Vision module đúng**:

1. **Cài đặt dependencies trên host** (đã làm ở Bước 2)

2. **Chạy Vision module riêng biệt** (trong terminal mới):

```powershell
# Từ thư mục src/
cd D:\AI_Final\Final_AI_Project\src

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Chạy Vision module (sẽ kết nối với Backend container qua API)
python -m vision.vision_app
```

3. **Hoặc tích hợp Vision vào Backend** (nếu Backend chạy trên host, không phải container):

```powershell
# Chạy main.py trên host (không dùng Docker)
python main.py
```

**Kiến trúc đề xuất**:
- **Backend + Chatbot**: Chạy trong Docker container (ổn định, dễ deploy)
- **Vision Module**: Chạy native trên host (truy cập webcam trực tiếp, ổn định hơn)
- **Kết nối**: Vision module gửi dữ liệu tới Backend container qua HTTP/WebSocket API

#### Troubleshooting Docker

> **⚠️ Lưu ý**: Do đang trong giai đoạn thử nghiệm, có thể gặp các lỗi không mong đợi. Vui lòng báo cáo issues để chúng tôi cải thiện.

##### Lỗi: `GOOGLE_API_KEY is not set` trong container

**Giải pháp**: 
- Kiểm tra file `src/chatbot/.env` tồn tại và có `GOOGLE_API_KEY`
- Xác nhận Docker Compose đọc đúng file: `env_file: - src/chatbot/.env`
- Kiểm tra logs: `docker-compose logs backend`

##### Lỗi: `Port 5000 is already allocated`

**Giải pháp**: 
- Đóng ứng dụng khác đang dùng port 5000
- Hoặc thay đổi port trong `docker-compose.yml`:
  ```yaml
  ports:
    - "8080:5000"  # Host:Container
  ```

##### Lỗi: `Cannot connect to ChromaDB` hoặc `Permission denied`

**Giải pháp**: 
- Kiểm tra quyền thư mục `src/data/`:
  ```bash
  # Linux/Mac:
  chmod -R 755 src/data/
  
  # Windows: Kiểm tra quyền trong Properties > Security
  ```
- Hoặc xóa và rebuild ChromaDB trong container:
  ```bash
  docker-compose exec backend rm -rf /app/src/data/chroma_db
  ```

##### Lỗi: `Package 'libgl1-mesa-glx' has no installation candidate` khi build

**Giải pháp**: 
- Đã được sửa trong Dockerfile (thay bằng `libgl1`)
- Nếu vẫn gặp lỗi, thử rebuild: `docker-compose build --no-cache`

##### Container không start hoặc crash ngay

**Giải pháp**: 
- Xem logs chi tiết: `docker-compose logs backend`
- Kiểm tra health check: `docker-compose ps`
- Rebuild image: `docker-compose build --no-cache`
- Kiểm tra file `.env` có đúng format không

##### Vision module không hoạt động khi chạy trong container

**Giải pháp**: 
- **Đây là hành vi mong đợi** - Vision module không được container hóa
- Cài đặt và chạy Vision module trên host (xem Bước 2)
- Vision module sẽ kết nối với Backend container qua API

##### Image quá lớn hoặc build chậm

**Giải pháp**: 
- Dockerfile đã dùng multi-stage build để tối ưu
- Lần đầu build sẽ chậm (download dependencies, ~5-10 phút)
- Lần sau sẽ nhanh hơn nhờ Docker cache
- Kiểm tra `.dockerignore` đã loại trừ các file lớn chưa

#### Development Mode (Hot Reload)

Để phát triển với code reload tự động, uncomment dòng này trong `docker-compose.yml`:

```yaml
volumes:
  - ./src/data:/app/src/data
  - ./src:/app/src  # Uncomment để mount source code
```

Sau đó restart: `docker-compose restart backend`

> ⚠️ **Lưu ý**: 
> - Development mode có thể chậm hơn do file I/O overhead
> - Chỉ áp dụng cho Backend/Chatbot code, không áp dụng cho Vision module
> - Vision module vẫn cần chạy trên host để truy cập webcam

#### Tóm tắt kiến trúc Docker

```
┌─────────────────────────────────────────┐
│  Docker Container (Backend + Chatbot)  │
│  - Flask API Server                     │
│  - LangGraph Chatbot                    │
│  - ChromaDB Vector Store               │
│  - Port: 5000                           │
└─────────────────────────────────────────┘
              ↑ HTTP/WebSocket
              │
┌─────────────────────────────────────────┐
│  Host Machine (Vision Module)           │
│  - OpenCV + MediaPipe                  │
│  - Webcam Access                        │
│  - Eye Tracking, Posture Analysis      │
│  - Chạy native (không container)       │
└─────────────────────────────────────────┘
```

**Lợi ích**:
- Backend/Chatbot: Dễ deploy, nhất quán giữa các môi trường
- Vision: Truy cập hardware trực tiếp, ổn định hơn

**Hạn chế**:
- Cần cài đặt Python dependencies trên host cho Vision module
- Không thể chạy hoàn toàn trong container (do Vision module)

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

