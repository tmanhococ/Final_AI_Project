## Chatbot Health Care – Adaptive RAG (`src/chatbot`)

Chatbot này là phần **RAG + Multi-node LangGraph** tách biệt khỏi hệ thống `vision` (camera). Toàn bộ kiến trúc bám theo tài liệu `RAG_Design_Docs_V2.md`: có guardrails, contextualize, CSV agent, retriever, grader, generator, rewriter và memory theo `thread_id`.

---

### 1. Chuẩn bị môi trường

- **Python**: 3.10+ (khuyến nghị 3.12).
- Khởi tạo virtualenv riêng cho chatbot:

```powershell
cd D:\AI_Final\Final_AI_Project
python -m venv .venv_chatbot
.\.venv_chatbot\Scripts\Activate.ps1
```

- Cài thư viện:

```powershell
pip install -r src\chatbot\requirements.txt
```

---

### 2. Thiết lập `GOOGLE_API_KEY` và `.env`

1. Tạo file `src/chatbot/.env` (ưu tiên) hoặc `.env` ở project root.
2. Thêm nội dung tối thiểu:

```env
GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"

# (không bắt buộc) Bật tracing LangSmith để debug:
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="YOUR_LANGSMITH_KEY_HERE"
LANGCHAIN_PROJECT="health-care-chatbot-student"

# (tùy chọn) override cấu hình mặc định
# LLM_MODEL_NAME="gemini-2.5-flash"
# EMBEDDING_MODEL_NAME="models/embedding-001"
# CHROMA_PERSIST_DIRECTORY="./data/chroma_db"
# CSV_FILE_PATH="./data/logs/user_health_log.csv"
# CHUNK_SIZE=1000
# CHUNK_OVERLAP=200
# K_RETRIEVAL=3
# MAX_RETRIES=3
```

> Lưu ý: code autoload `.env` trong `src/chatbot/` trước, nếu không có mới fallback sang `.env` ở root.

---

### 3. Các module chính (bức tranh kiến trúc)

- **Cấu hình & state**
  - `config.py`: class `ChatbotConfig` + instance `CHATBOT_CONFIG` (đọc `.env`, validate cấu hình).
  - `state.py`: `GraphState` (TypedDict) – định nghĩa state chung cho LangGraph (messages, question, generation, intent, sub_queries, context, retry_count, answer_valid).

- **Kho dữ liệu & tools**
  - `tools/vector_store.py`:
    - Load các file `.txt` trong `data/medical_docs/`.
    - Tạo / load **Chroma** vector DB bằng `GoogleGenerativeAIEmbeddings` (embedding-001).
  - `tools/csv_loader.py`:
    - Đọc `src/data/summary.csv` (log sức khỏe) vào `pandas.DataFrame`.
    - Tạo **Pandas DataFrame Agent** cho phân tích thống kê (dựa trên LLM).

- **Các node (LangGraph)**
  - `nodes/chat_utils.py`:
    - `detect_intent_from_text`, `guardrails_node`: phân loại `social` vs `health`.
    - `social_response_node`: trả lời xã giao nhanh (dùng LLM).
    - `contextualize_question`, `contextualize_node`: viết lại câu hỏi dựa vào lịch sử chat.
  - `nodes/query_analysis.py`:
    - `analyze_query_node`: xác định intent (`realtime_data`, `chunked_data`, `both`, `fall_back`) + `sub_queries`.
  - `nodes/csv_node.py`:
    - `csv_analyst_node`: gọi Pandas agent, append kết quả vào `context`.
  - `nodes/retriever_node.py`:
    - `medical_retriever_node`: truy vấn Chroma, append document text vào `context`.
  - `nodes/grader_node.py`:
    - `doc_grader_node`: lọc context theo độ liên quan.
    - `answer_grader_node`: đánh giá câu trả lời có “ổn” không.
  - `nodes/generator_node.py`:
    - `generator_node`: sinh câu trả lời cuối cùng từ `context` + `reformulated_question`.
  - `nodes/rewriter_node.py`:
    - `rewriter_node`: viết lại query khi answer chưa tốt, tăng `retry_count`.

- **Lắp ráp graph & runtime**
  - `graph.py`:
    - Hàm `build_graph(llm, vector_store, csv_agent, max_retries)`:
      - Flow: `guardrails -> (social_bot | contextualize) -> query_analysis -> csv_node/retriever_node/both/no_rag -> doc_grader -> generator -> answer_grader -> (END | rewriter loop)`.
      - Dùng `MemorySaver` làm checkpointer (ghi nhớ theo `thread_id`).
  - `llm_factory.py`:
    - `create_production_llm()`: tạo `ChatGoogleGenerativeAI(model="gemini-2.5-flash")`.
    - `create_production_embeddings()`: tạo `GoogleGenerativeAIEmbeddings(model="models/embedding-001")`.
  - `app_runtime.py`:
    - `create_chatbot_app()`:
      - Validate config.
      - Tạo LLM, vector store, CSV agent.
      - Gọi `build_graph(...)` → trả về LangGraph app hoàn chỉnh.
  - `chat_interface.py`:
    - Hàm public:

```python
from src.chatbot.chat_interface import chat_interface

answer = chat_interface(
    "Tôi bị mỏi mắt khi làm việc với máy tính, tôi nên làm gì?",
    thread_id="user1",
)
print(answer)
```

    - Nếu không truyền `app`, nó tự gọi `create_chatbot_app()` và dùng LLM thật.

---

### 4. Cách chạy test (pytest) kiểm tra từng lớp

Khi đã activate venv:

```powershell
cd D:\AI_Final\Final_AI_Project
pytest -q -s
```

Một số file test quan trọng:

- `test_config.py`: test đọc env + validate cấu hình.
- `test_tools.py`: test `vector_store` & `csv_loader` với Embedding/LLM giả (không tốn quota).
- `test_nodes.py`: test từng node (guardrails, contextualize, csv_node, retriever_node, grader, generator, rewriter).
- `test_graph.py`: test toàn bộ LangGraph (route social + route health).
- `test_chat_interface.py`: test `chat_interface` end-to-end với graph Dummy.
- `test_google_api_key.py`: script **kiểm tra trực tiếp GOOGLE_API_KEY với Gemini thật**:

```powershell
python -m src.chatbot.test_google_api_key
```

> Khi chạy với `-s`, các test sẽ in rõ input/output (câu hỏi, context, câu trả lời giả…) ra console cho bạn dễ kiểm tra.

---

### 5. Tích hợp chatbot vào GUI / Desktop App

Trong code GUI (PyQt, Tkinter, hoặc app main của bạn), chỉ cần wrap `chat_interface`:

```python
from src.chatbot.chat_interface import chat_interface

def ask_bot(user_text: str, user_id: str = "local_user_1") -> str:
    return chat_interface(user_text, thread_id=user_id)

if __name__ == "__main__":
    while True:
        q = input("Bạn: ")
        if not q.strip():
            break
        a = ask_bot(q)
        print("Bot:", a)
```

- `thread_id` giúp **ghi nhớ lịch sử hội thoại** cho từng user/phiên chat.
- Bạn có thể map `thread_id` = ID user trong hệ thống login của app.

---

### 6. Gợi ý mở rộng / chỉnh sửa

- Nếu muốn đổi model:
  - Sửa `LLM_MODEL_NAME` trong `.env` hoặc `config.py` (mặc định: `gemini-2.5-flash`).
- Nếu thêm tài liệu y tế mới:
  - Thêm `.txt` vào `data/medical_docs/`.
  - Xóa `data/chroma_db` nếu muốn build lại từ đầu (hoặc dùng `force_rebuild=True` trong `build_or_load_medical_vector_store`).
- Khi chỉnh sửa graph/nodes:
  - Sửa file trong `nodes/*.py` hoặc `graph.py`.
  - Chạy lại test tương ứng:

```powershell
pytest src\chatbot\test_nodes.py src\chatbot\test_graph.py -q -s
```



