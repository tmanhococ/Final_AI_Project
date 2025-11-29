# Tài liệu Chi tiết: Healthcare Chatbot - Adaptive RAG System

## Mục lục

1. [Tổng quan](#tổng-quan)
2. [Cấu trúc thư mục](#cấu-trúc-thư-mục)
3. [Cấu hình và State](#cấu-hình-và-state)
4. [Tools và Utilities](#tools-và-utilities)
5. [Nodes (LangGraph)](#nodes-langgraph)
6. [Graph Assembly](#graph-assembly)
7. [Runtime và Interface](#runtime-và-interface)
8. [Ví dụ sử dụng](#ví-dụ-sử-dụng)
9. [Testing](#testing)

---

## Tổng quan

Chatbot Healthcare là hệ thống **Adaptive RAG (Retrieval-Augmented Generation)** sử dụng LangGraph để xây dựng workflow đa bước cho việc trả lời câu hỏi về sức khỏe mắt và tư thế.

### Kiến trúc tổng thể

```
User Input
    ↓
[Guardrails] → Social? → [Social Bot] → END
    ↓ Health
[Contextualize] → [Query Analysis] → Intent Detection
    ↓
    ├─→ CSV Only → [CSV Node] → [Generator] → [Answer Grader] → END
    ├─→ Retriever Only → [Retriever] → [Doc Grader] → [Generator] → [Answer Grader] → END
    ├─→ Both → [CSV Node] → [Retriever] → [Doc Grader] → [Generator] → [Answer Grader] → END
    └─→ Fallback → [Generator] → [Answer Grader] → END
                                    ↓ (if invalid)
                                [Rewriter] → [Query Analysis] (retry loop)
```

### Luồng dữ liệu

1. **Guardrails**: Phân loại câu hỏi thành `social` (chào hỏi) hoặc `health` (câu hỏi y tế)
2. **Contextualize**: Viết lại câu hỏi dựa trên lịch sử hội thoại
3. **Query Analysis**: Phân tích intent và tạo sub-queries
4. **RAG Nodes**: 
   - **CSV Node**: Truy vấn dữ liệu thống kê từ `summary.csv`
   - **Retriever Node**: Tìm kiếm documents từ ChromaDB
5. **Grader**: Lọc context và đánh giá chất lượng câu trả lời
6. **Generator**: Sinh câu trả lời cuối cùng
7. **Rewriter**: Viết lại query nếu câu trả lời không hợp lệ (retry loop)

---

## Cấu trúc thư mục

```
src/chatbot/
├── config.py              # Cấu hình chatbot (API keys, model names, paths)
├── state.py               # Định nghĩa GraphState (TypedDict)
├── llm_factory.py         # Factory tạo LLM và embeddings
├── app_runtime.py         # Khởi tạo chatbot app hoàn chỉnh
├── chat_interface.py      # Interface công khai cho người dùng
├── graph.py               # Lắp ráp LangGraph workflow
├── app.py                 # CLI application
├── nodes/                 # Các node trong LangGraph
│   ├── chat_utils.py      # Guardrails, Social Bot, Contextualize
│   ├── query_analysis.py  # Phân tích intent và tạo sub-queries
│   ├── csv_node.py        # Truy vấn CSV data
│   ├── retriever_node.py  # Truy vấn vector store
│   ├── grader_node.py     # Lọc context và đánh giá answer
│   ├── generator_node.py  # Sinh câu trả lời
│   └── rewriter_node.py   # Viết lại query khi retry
├── tools/                 # Utilities cho data loading
│   ├── vector_store.py    # ChromaDB vector store utilities
│   └── csv_loader.py      # Pandas DataFrame agent utilities
└── test/                  # Test files
    ├── test_config.py
    ├── test_tools.py
    ├── test_nodes.py
    ├── test_graph.py
    ├── test_chat_interface.py
    └── test_fixes.py
```

---

## Cấu hình và State

### `config.py`

#### Class: `ChatbotConfig`

**Mô tả**: Dataclass immutable chứa toàn bộ cấu hình chatbot (API keys, model names, paths, RAG parameters).

**Thuộc tính**:

| Thuộc tính | Kiểu | Mô tả |
|------------|------|-------|
| `google_api_key` | `Optional[str]` | API key cho Google Gemini (bắt buộc) |
| `langsmith_api_key` | `Optional[str]` | API key cho LangSmith tracing (tùy chọn) |
| `langsmith_project` | `Optional[str]` | Tên project LangSmith (tùy chọn) |
| `langsmith_endpoint` | `Optional[str]` | Endpoint LangSmith (tùy chọn) |
| `langsmith_tracing` | `bool` | Bật/tắt LangSmith tracing |
| `llm_model_name` | `str` | Tên model LLM (mặc định: `"gemini-2.5-flash"`) |
| `embedding_model_name` | `str` | Tên model embedding (mặc định: `"models/embedding-001"`) |
| `chroma_persist_directory` | `Path` | Đường dẫn lưu ChromaDB (mặc định: `data/chroma_db`) |
| `csv_file_path` | `Path` | Đường dẫn file CSV (mặc định: `src/data/logs/user_health_log.csv`) |
| `chunk_size` | `int` | Kích thước chunk khi split documents (mặc định: 1000) |
| `chunk_overlap` | `int` | Độ overlap giữa các chunks (mặc định: 200) |
| `k_retrieval` | `int` | Số documents retrieve từ vector store (mặc định: 3) |
| `max_retries` | `int` | Số lần retry tối đa khi answer không hợp lệ (mặc định: 3) |

**Methods**:

##### `from_env() -> ChatbotConfig`

**Mô tả**: Tạo `ChatbotConfig` từ biến môi trường trong file `.env`.

**Environment Variables**:
- `GOOGLE_API_KEY`: API key cho Google Gemini (bắt buộc)
- `LANGCHAIN_API_KEY`: API key cho LangSmith (tùy chọn)
- `LANGCHAIN_PROJECT`: Tên project LangSmith (tùy chọn)
- `LANGCHAIN_ENDPOINT`: Endpoint LangSmith (tùy chọn)
- `LANGCHAIN_TRACING_V2`: `"true"/"false"` bật tắt tracing
- `LLM_MODEL_NAME`: Tên model LLM (mặc định: `"gemini-2.5-flash"`)
- `EMBEDDING_MODEL_NAME`: Tên model embedding (mặc định: `"models/embedding-001"`)
- `CHROMA_PERSIST_DIRECTORY`: Đường dẫn lưu ChromaDB
- `CSV_FILE_PATH`: Đường dẫn file CSV
- `CHUNK_SIZE`: Kích thước chunk (mặc định: 1000)
- `CHUNK_OVERLAP`: Độ overlap (mặc định: 200)
- `K_RETRIEVAL`: Số documents retrieve (mặc định: 3)
- `MAX_RETRIES`: Số lần retry tối đa (mặc định: 3)

**Returns**: `ChatbotConfig` instance

**Ví dụ**:
```python
from src.chatbot.config import CHATBOT_CONFIG

# CHATBOT_CONFIG đã được tạo tự động từ .env
print(CHATBOT_CONFIG.llm_model_name)  # "gemini-2.5-flash"
```

##### `validate() -> None`

**Mô tả**: Validate cấu hình và raise `ValueError` nếu không hợp lệ.

**Raises**:
- `ValueError`: Nếu thiếu `GOOGLE_API_KEY` hoặc các tham số số học không hợp lệ

**Ví dụ**:
```python
CHATBOT_CONFIG.validate()  # Raise ValueError nếu GOOGLE_API_KEY chưa set
```

##### `has_langsmith -> bool` (property)

**Mô tả**: Kiểm tra xem LangSmith đã được cấu hình đầy đủ chưa.

**Returns**: `True` nếu có `langsmith_api_key` và `langsmith_project`

---

### `state.py`

#### Class: `GraphState`

**Mô tả**: TypedDict định nghĩa state chung cho toàn bộ LangGraph workflow.

**Fields**:

| Field | Kiểu | Mô tả |
|-------|------|-------|
| `messages` | `Annotated[List[BaseMessage], add_messages]` | Lịch sử hội thoại (User + AI), tự động merge khi update |
| `original_question` | `str` | Câu hỏi gốc mới nhất của user |
| `reformulated_question` | `str` | Câu hỏi đã được contextualize (viết lại) |
| `generation` | `str` | Câu trả lời cuối cùng của chatbot |
| `analyzed_intent` | `Literal["realtime_data", "chunked_data", "both", "fall_back"]` | Intent đã phân tích |
| `sub_queries` | `List[str]` | Danh sách câu hỏi con để truy vấn CSV/retriever |
| `context` | `List[str]` | Context đã được merge và lọc (csv_context + filtered_docs) |
| `csv_context` | `NotRequired[List[str]]` | Kết quả từ CSV node (tùy chọn) |
| `doc_context` | `NotRequired[List[str]]` | Documents từ retriever node (tùy chọn) |
| `route` | `NotRequired[Literal["social", "health"]]` | Route từ guardrails (tùy chọn) |
| `retry_count` | `int` | Số lần đã retry (bắt đầu từ 0) |
| `answer_valid` | `bool` | Cờ đánh dấu câu trả lời có hợp lệ không |

**Lưu ý**: 
- `messages` dùng `add_messages` reducer để tự động merge, tránh duplicate
- Các node chỉ trả về fields được modify, không trả về toàn bộ state để tránh duplicate messages

**Ví dụ**:
```python
from src.chatbot.state import GraphState
from langchain_core.messages import HumanMessage

state: GraphState = {
    "messages": [HumanMessage(content="CVS là gì?")],
    "original_question": "CVS là gì?",
    "reformulated_question": "",
    "generation": "",
    "analyzed_intent": "fall_back",
    "sub_queries": [],
    "context": [],
    "retry_count": 0,
    "answer_valid": True,
}
```

---

## Tools và Utilities

### `tools/vector_store.py`

#### Function: `get_default_paths(config: ChatbotConfig = CHATBOT_CONFIG) -> MedicalVectorStorePaths`

**Mô tả**: Trả về các đường dẫn mặc định cho medical documents và Chroma persistence.

**Parameters**:
- `config`: ChatbotConfig instance (mặc định: `CHATBOT_CONFIG`)

**Returns**: `MedicalVectorStorePaths` với:
- `docs_dir`: `Path` - Thư mục chứa medical documents (`src/data/medical_docs`)
- `persist_dir`: `Path` - Thư mục lưu ChromaDB (từ `config.chroma_persist_directory`)

**Ví dụ**:
```python
from src.chatbot.tools.vector_store import get_default_paths

paths = get_default_paths()
print(paths.docs_dir)  # Path("src/data/medical_docs")
print(paths.persist_dir)  # Path("data/chroma_db")
```

---

#### Function: `load_medical_documents(docs_dir: Path) -> List[Document]`

**Mô tả**: Load tất cả file `.txt` trong thư mục `docs_dir` thành LangChain `Document` objects.

**Parameters**:
- `docs_dir`: `Path` - Thư mục chứa medical documents

**Returns**: `List[Document]` - Danh sách documents đã được load

**Raises**:
- `FileNotFoundError`: Nếu `docs_dir` không tồn tại
- `ValueError`: Nếu không tìm thấy file `.txt` nào

**Ví dụ**:
```python
from pathlib import Path
from src.chatbot.tools.vector_store import load_medical_documents

docs = load_medical_documents(Path("src/data/medical_docs"))
print(f"Loaded {len(docs)} documents")
```

---

#### Class: `SafeGoogleEmbeddings`

**Mô tả**: Wrapper quanh `GoogleGenerativeAIEmbeddings` với fallback local embedding khi hết quota (429 error).

**Methods**:

##### `__init__(model_name: str, api_key: str) -> None`

**Parameters**:
- `model_name`: Tên model embedding (ví dụ: `"models/embedding-001"`)
- `api_key`: Google API key

##### `embed_documents(texts: List[str]) -> List[List[float]]`

**Mô tả**: Embed danh sách documents. Nếu Google API lỗi, fallback sang local hash-based embeddings.

**Parameters**:
- `texts`: Danh sách texts cần embed

**Returns**: `List[List[float]]` - Danh sách embedding vectors

##### `embed_query(text: str) -> List[float]`

**Mô tả**: Embed một query text. Nếu Google API lỗi, fallback sang local embedding.

**Parameters**:
- `text`: Query text cần embed

**Returns**: `List[float]` - Embedding vector

**Lưu ý**: Local embeddings không chính xác về mặt semantic, chỉ dùng để tránh crash.

---

#### Function: `create_huggingface_embeddings() -> Embeddings`

**Mô tả**: Tạo HuggingFace embeddings instance (local, miễn phí, không có quota limit).

**Model**: `sentence-transformers/all-MiniLM-L6-v2` (dimension 384, hỗ trợ đa ngôn ngữ)

**Returns**: `Embeddings` - LangChain embeddings implementation

**Raises**:
- `ImportError`: Nếu `langchain-huggingface` hoặc `langchain-community` chưa được cài

**Ví dụ**:
```python
from src.chatbot.tools.vector_store import create_huggingface_embeddings

embeddings = create_huggingface_embeddings()
query_embedding = embeddings.embed_query("CVS là gì?")
print(f"Dimension: {len(query_embedding)}")  # 384
```

---

#### Function: `create_google_embeddings(config: ChatbotConfig = CHATBOT_CONFIG) -> Embeddings`

**Mô tả**: Tạo Google Generative AI embeddings với fallback local nếu hết quota.

**Parameters**:
- `config`: ChatbotConfig instance (mặc định: `CHATBOT_CONFIG`)

**Returns**: `SafeGoogleEmbeddings` instance

**Raises**:
- `ValueError`: Nếu `GOOGLE_API_KEY` chưa được cấu hình

**Lưu ý**: Khuyến nghị dùng `create_huggingface_embeddings()` để tránh quota limit.

---

#### Function: `build_chroma_from_documents(documents: Iterable[Document], embeddings: Embeddings, persist_directory: Optional[Path] = None) -> Chroma`

**Mô tả**: Xây dựng Chroma vector store từ documents và embeddings.

**Parameters**:
- `documents`: Iterable of LangChain `Document` objects
- `embeddings`: Embeddings implementation
- `persist_directory`: Optional directory để persist ChromaDB (nếu `None`, DB là in-memory)

**Returns**: `Chroma` - Vector store đã được xây dựng

**Raises**:
- `ValueError`: Nếu documents rỗng

**Ví dụ**:
```python
from src.chatbot.tools.vector_store import (
    load_medical_documents,
    create_huggingface_embeddings,
    build_chroma_from_documents,
)
from pathlib import Path

docs = load_medical_documents(Path("src/data/medical_docs"))
embeddings = create_huggingface_embeddings()
vs = build_chroma_from_documents(docs, embeddings, persist_directory=Path("data/chroma_db"))
```

---

#### Function: `build_or_load_medical_vector_store(config: ChatbotConfig = CHATBOT_CONFIG, *, paths: Optional[MedicalVectorStorePaths] = None, embeddings: Optional[Embeddings] = None, force_rebuild: bool = False, use_huggingface: bool = True) -> Chroma`

**Mô tả**: Build hoặc load Chroma vector store cho medical documents.

**Logic**:
- Nếu `force_rebuild=True` hoặc `persist_dir` chưa tồn tại: load documents và build mới
- Ngược lại: load Chroma từ `persist_dir`
- Nếu có lỗi load (dimension mismatch, version mismatch): tự động rebuild

**Parameters**:
- `config`: ChatbotConfig instance (mặc định: `CHATBOT_CONFIG`)
- `paths`: Optional custom paths (nếu `None`, dùng `get_default_paths`)
- `embeddings`: Optional embeddings (nếu `None`, tạo từ `use_huggingface` flag)
- `force_rebuild`: Nếu `True`, luôn rebuild DB từ documents
- `use_huggingface`: Nếu `True` (mặc định), dùng HuggingFace embeddings (local, miễn phí)

**Returns**: `Chroma` - Vector store sẵn sàng cho truy vấn

**Ví dụ**:
```python
from src.chatbot.tools.vector_store import build_or_load_medical_vector_store
from src.chatbot.config import CHATBOT_CONFIG

# Dùng HuggingFace embeddings (mặc định)
vs = build_or_load_medical_vector_store(CHATBOT_CONFIG, force_rebuild=False)

# Dùng Google embeddings
vs = build_or_load_medical_vector_store(
    CHATBOT_CONFIG,
    use_huggingface=False,
    force_rebuild=True
)
```

---

### `tools/csv_loader.py`

#### Function: `get_summary_csv_path(config: ChatbotConfig = CHATBOT_CONFIG) -> Path`

**Mô tả**: Trả về đường dẫn mặc định đến `summary.csv`.

**Parameters**:
- `config`: ChatbotConfig instance (mặc định: `CHATBOT_CONFIG`)

**Returns**: `Path` - Đường dẫn đến `summary.csv` (mặc định: `src/data/summary.csv`)

---

#### Function: `load_summary_dataframe(config: ChatbotConfig = CHATBOT_CONFIG) -> pd.DataFrame`

**Mô tả**: Load `summary.csv` thành pandas DataFrame.

**Parameters**:
- `config`: ChatbotConfig instance (mặc định: `CHATBOT_CONFIG`)

**Returns**: `pd.DataFrame` - DataFrame chứa dữ liệu từ `summary.csv`

**Raises**:
- `FileNotFoundError`: Nếu file CSV không tồn tại

**Ví dụ**:
```python
from src.chatbot.tools.csv_loader import load_summary_dataframe

df = load_summary_dataframe()
print(df.head())
```

---

#### Function: `create_summary_agent(df: pd.DataFrame, llm: BaseLanguageModel, verbose: bool = False) -> AgentExecutor`

**Mô tả**: Tạo Pandas DataFrame Agent để phân tích dữ liệu trong `df`.

**Parameters**:
- `df`: `pd.DataFrame` - DataFrame chứa dữ liệu health logs
- `llm`: `BaseLanguageModel` - LLM instance để agent sử dụng
- `verbose`: `bool` - Bật/tắt verbose output (mặc định: `False`)

**Returns**: `AgentExecutor` - LangChain agent có thể gọi `.invoke({"input": question})`

**Raises**:
- `ModuleNotFoundError`: Nếu `langchain-experimental` chưa được cài

**Lưu ý**: Agent được cấu hình với `handle_parsing_errors=True` để xử lý lỗi parsing từ LLM.

**Ví dụ**:
```python
from src.chatbot.tools.csv_loader import load_summary_dataframe, create_summary_agent
from src.chatbot.llm_factory import create_production_llm

df = load_summary_dataframe()
llm = create_production_llm()
agent = create_summary_agent(df, llm, verbose=False)

result = agent.invoke({"input": "Trung bình thời lượng các phiên đo là bao nhiêu phút?"})
print(result["output"])
```

---

## Nodes (LangGraph)

### `nodes/chat_utils.py`

#### Function: `detect_intent_from_text(text: str) -> Literal["social", "health"]`

**Mô tả**: Phân loại câu hỏi thành `social` (chào hỏi/xã giao) hoặc `health` (câu hỏi y tế) dựa trên heuristic keyword matching.

**Parameters**:
- `text`: Latest user message content

**Returns**: `"social"` nếu là câu chào hỏi, `"health"` nếu là câu hỏi y tế

**Logic**:
- Exact match: `"hi"`, `"hey"`, `"hello"` → `"social"`
- Keyword matching: tìm các từ khóa xã giao trong text → `"social"`
- Mặc định: `"health"`

**Ví dụ**:
```python
from src.chatbot.nodes.chat_utils import detect_intent_from_text

intent1 = detect_intent_from_text("Hi, bạn khỏe không?")  # "social"
intent2 = detect_intent_from_text("CVS là gì?")  # "health"
```

---

#### Function: `guardrails_node(state: StateDict) -> Dict[str, object]`

**Mô tả**: Node guardrails - phân loại câu hỏi thành social hoặc health. Đây là entry point đầu tiên của graph.

**Parameters**:
- `state`: GraphState với field `messages` chứa lịch sử hội thoại

**Returns**: `{"route": "social" | "health"}`

**Raises**:
- `ValueError`: Nếu `messages` rỗng hoặc message cuối không phải `HumanMessage`

**Ví dụ**:
```python
from src.chatbot.nodes.chat_utils import guardrails_node
from langchain_core.messages import HumanMessage

state = {"messages": [HumanMessage(content="Hi")]}
result = guardrails_node(state)
print(result["route"])  # "social"
```

---

#### Function: `social_response_node(state: StateDict, llm: BaseLanguageModel) -> Dict[str, object]`

**Mô tả**: Node trả lời các câu chào hỏi/xã giao (social bot).

**Parameters**:
- `state`: GraphState với field `messages`
- `llm`: LLM instance để sinh câu trả lời

**Returns**: `{"generation": str}` - Câu trả lời đã được sinh

**Raises**:
- `ValueError`: Nếu `messages` rỗng

**Ví dụ**:
```python
from src.chatbot.nodes.chat_utils import social_response_node
from src.chatbot.llm_factory import create_production_llm
from langchain_core.messages import HumanMessage

state = {"messages": [HumanMessage(content="Hi")]}
llm = create_production_llm()
result = social_response_node(state, llm)
print(result["generation"])  # "Chào bạn!"
```

---

#### Function: `contextualize_question(messages: Iterable[BaseMessage], original_question: str, llm: BaseLanguageModel) -> str`

**Mô tả**: Viết lại câu hỏi dựa trên lịch sử hội thoại. **QUAN TRỌNG**: Chỉ viết lại RIÊNG câu hỏi mới nhất, không tổng hợp các câu hỏi trước.

**Parameters**:
- `messages`: Full chat history (user + assistant) để cung cấp context
- `original_question`: Câu hỏi mới nhất của user cần được viết lại
- `llm`: LLM instance để rewrite

**Returns**: `str` - Câu hỏi đã được viết lại (CHỈ MỘT CÂU DUY NHẤT)

**Ví dụ**:
```python
from src.chatbot.nodes.chat_utils import contextualize_question
from langchain_core.messages import HumanMessage, AIMessage
from src.chatbot.llm_factory import create_production_llm

messages = [
    HumanMessage(content="Tôi bị mỏi mắt"),
    AIMessage(content="Bạn có thể thử quy tắc 20-20-20"),
    HumanMessage(content="Quy tắc đó là gì?"),
]
llm = create_production_llm()
reformulated = contextualize_question(messages, "Quy tắc đó là gì?", llm)
print(reformulated)  # "Quy tắc 20-20-20 là gì?" (chỉ viết lại câu hỏi mới nhất)
```

---

#### Function: `contextualize_node(state: StateDict, llm: BaseLanguageModel) -> Dict[str, object]`

**Mô tả**: Node contextualize - viết lại câu hỏi dựa trên lịch sử hội thoại.

**Parameters**:
- `state`: GraphState với fields `messages` và `original_question`
- `llm`: LLM instance để viết lại câu hỏi

**Returns**: `{"reformulated_question": str}`

**Raises**:
- `ValueError`: Nếu `messages` rỗng hoặc `original_question` rỗng

---

### `nodes/query_analysis.py`

#### Function: `heuristic_analyze_intent(question: str) -> IntentLiteral`

**Mô tả**: Phân loại intent dựa trên keyword matching.

**Parameters**:
- `question`: Câu hỏi đã được contextualize

**Returns**: 
- `"realtime_data"`: Câu hỏi về dữ liệu thống kê/logs (dùng CSV agent)
- `"chunked_data"`: Câu hỏi về kiến thức y tế (dùng vector retriever)
- `"both"`: Câu hỏi cần cả CSV và documents
- `"chunked_data"`: Mặc định (ưu tiên khai thác docs)

**Keywords**:
- CSV keywords: `"log"`, `"session"`, `"thống kê"`, `"summary"`, `"phiên đo"`, `"thời lượng"`, `"trung bình"`, `"tổng"`, `"số lượng"`, etc.
- Doc keywords: `"bệnh"`, `"triệu chứng"`, `"nguyên nhân"`, `"điều trị"`, `"hội chứng"`, `"mỏi mắt"`, `"là gì"`, `"giải thích"`, `"bác sĩ"`, `"là ai"`, etc.

**Ví dụ**:
```python
from src.chatbot.nodes.query_analysis import heuristic_analyze_intent

intent1 = heuristic_analyze_intent("Trung bình thời lượng các phiên đo là bao nhiêu?")  # "realtime_data"
intent2 = heuristic_analyze_intent("CVS là gì?")  # "chunked_data"
intent3 = heuristic_analyze_intent("Thống kê thời lượng và giải thích CVS là gì")  # "both"
```

---

#### Function: `analyze_query_node(state: StateDict, llm: Optional[BaseLanguageModel] = None) -> Dict[str, object]`

**Mô tả**: Node phân tích câu hỏi - xác định intent và tạo sub-queries.

**QUAN TRỌNG**: Node này RESET `sub_queries` (không append) để tránh tích lũy.

**Parameters**:
- `state`: GraphState với field `reformulated_question`
- `llm`: Optional LLM (hiện tại không dùng, có thể mở rộng)

**Returns**: 
```python
{
    "analyzed_intent": IntentLiteral,
    "sub_queries": List[str]  # RESET, không append
}
```

**Raises**:
- `ValueError`: Nếu `reformulated_question` rỗng

---

### `nodes/csv_node.py`

#### Function: `csv_analyst_node(state: StateDict, agent: SupportsInvoke) -> Dict[str, object]`

**Mô tả**: Node truy vấn CSV data qua Pandas DataFrame Agent.

**QUAN TRỌNG**: Node này RESET `csv_context` (không append vào context cũ).

**Parameters**:
- `state`: GraphState với field `sub_queries`
- `agent`: Pandas DataFrame Agent (có method `.invoke()`)

**Returns**: `{"csv_context": List[str]}` - Danh sách kết quả phân tích CSV (RESET)

**Raises**:
- `ValueError`: Nếu `sub_queries` rỗng

**Error Handling**:
- Bắt `ValueError` (parsing errors) và cố gắng extract raw LLM response từ error message
- Bắt `Exception` cho các lỗi khác (network, API quota, etc.)

**Ví dụ**:
```python
from src.chatbot.nodes.csv_node import csv_analyst_node
from src.chatbot.tools.csv_loader import create_summary_agent, load_summary_dataframe
from src.chatbot.llm_factory import create_production_llm

df = load_summary_dataframe()
llm = create_production_llm()
agent = create_summary_agent(df, llm)

state = {"sub_queries": ["Trung bình thời lượng là bao nhiêu?"]}
result = csv_analyst_node(state, agent)
print(result["csv_context"])  # ["Trung bình duration_minutes là 0.45 phút"]
```

---

### `nodes/retriever_node.py`

#### Function: `medical_retriever_node(state: StateDict, vector_store: Chroma) -> Dict[str, object]`

**Mô tả**: Node retrieve documents từ Chroma vector store.

**QUAN TRỌNG**: Node này RESET `doc_context` (không append vào context cũ).

**Parameters**:
- `state`: GraphState với field `sub_queries`
- `vector_store`: Chroma vector store instance

**Returns**: `{"doc_context": List[str]}` - Danh sách documents đã retrieve (RESET)

**Raises**:
- `ValueError`: Nếu `sub_queries` rỗng

**Lưu ý**: Sử dụng `k=3` (top 3 documents liên quan nhất) cho mỗi sub_query.

**Ví dụ**:
```python
from src.chatbot.nodes.retriever_node import medical_retriever_node
from src.chatbot.tools.vector_store import build_or_load_medical_vector_store
from src.chatbot.config import CHATBOT_CONFIG

vector_store = build_or_load_medical_vector_store(CHATBOT_CONFIG)
state = {"sub_queries": ["CVS là gì?"]}
result = medical_retriever_node(state, vector_store)
print(f"Retrieved {len(result['doc_context'])} document blocks")
```

---

### `nodes/grader_node.py`

#### Function: `document_relevance_grader(doc_text: str, question: str) -> bool`

**Mô tả**: Heuristic đơn giản kiểm tra document có liên quan đến câu hỏi không.

**Parameters**:
- `doc_text`: Nội dung của document
- `question`: Câu hỏi đã được contextualize

**Returns**: `True` nếu document liên quan (có ít nhất 1 từ khóa dài > 4 ký tự trùng), `False` nếu không

---

#### Function: `answer_quality_grader(answer: str, question: str, context: str) -> bool`

**Mô tả**: Heuristic đơn giản kiểm tra câu trả lời có "hợp lý" không.

**Parameters**:
- `answer`: Câu trả lời đã được sinh
- `question`: Câu hỏi đã được contextualize
- `context`: Context đã được sử dụng (không dùng trong logic hiện tại)

**Returns**: 
- `False` nếu answer quá ngắn (< 10 ký tự) hoặc chỉ lặp lại câu hỏi
- `True` nếu hợp lệ

---

#### Function: `doc_grader_node(state: StateDict) -> Dict[str, object]`

**Mô tả**: Node lọc documents không liên quan và merge `csv_context` + `filtered_docs` vào `context`.

**QUAN TRỌNG**: Node này RESET `context` (không append) để tránh tích lũy.

**Parameters**:
- `state`: GraphState với fields `doc_context`, `csv_context`, `reformulated_question`

**Returns**:
```python
{
    "doc_context": List[str],  # Docs đã lọc (chỉ giữ lại docs liên quan)
    "context": List[str]  # csv_context + filtered_docs (RESET)
}
```

**Raises**:
- `ValueError`: Nếu `reformulated_question` rỗng

**Logic**:
- Lọc `doc_context` bằng `document_relevance_grader`
- Merge `csv_context` (không lọc) + `filtered_docs` vào `context`

---

#### Function: `answer_grader_node(state: StateDict) -> Dict[str, object]`

**Mô tả**: Node đánh giá chất lượng câu trả lời để quyết định có cần retry không.

**Parameters**:
- `state`: GraphState với fields `generation`, `reformulated_question`, `context`

**Returns**: `{"answer_valid": bool}`

**Logic**:
- Gọi `answer_quality_grader` để kiểm tra
- Nếu `answer_valid=False` và `retry_count < max_retries`, graph sẽ quay lại `rewriter_node`

---

### `nodes/generator_node.py`

#### Function: `generator_node(state: StateDict, llm: BaseLanguageModel) -> Dict[str, object]`

**Mô tả**: Node sinh câu trả lời cuối cùng từ context và câu hỏi.

**QUAN TRỌNG**: Chỉ sử dụng câu hỏi mới nhất (`reformulated_question`), không tổng hợp các câu hỏi trước.

**Parameters**:
- `state`: GraphState với fields:
  - `reformulated_question`: Câu hỏi đã được contextualize (CHỈ câu hỏi mới nhất)
  - `context`: Context đã được merge từ `doc_grader` (ưu tiên)
  - `csv_context`, `doc_context`: Fallback nếu `context` rỗng
- `llm`: LLM instance để sinh câu trả lời

**Returns**: `{"generation": str}`

**Raises**:
- `ValueError`: Nếu `reformulated_question` rỗng

**Fallback Logic**:
- Nếu `context` rỗng, trả về message mặc định: "Xin lỗi, tôi không có đủ thông tin..."

**Ví dụ**:
```python
from src.chatbot.nodes.generator_node import generator_node
from src.chatbot.llm_factory import create_production_llm

state = {
    "reformulated_question": "CVS là gì?",
    "context": ["CVS là hội chứng thị giác máy tính..."],
}
llm = create_production_llm()
result = generator_node(state, llm)
print(result["generation"])
```

---

### `nodes/rewriter_node.py`

#### Function: `rewrite_query(question: str, last_answer: str, llm: BaseLanguageModel) -> str`

**Mô tả**: Dùng LLM để viết lại query tốt hơn khi câu trả lời trước không hợp lệ.

**Parameters**:
- `question`: Câu hỏi hiện tại cần được viết lại
- `last_answer`: Câu trả lời trước đó không hợp lệ
- `llm`: LLM instance để viết lại câu hỏi

**Returns**: `str` - Câu hỏi đã được viết lại, rõ ràng và cụ thể hơn

---

#### Function: `rewriter_node(state: StateDict, llm: BaseLanguageModel) -> Dict[str, object]`

**Mô tả**: Node viết lại query và tăng retry_count khi câu trả lời không hợp lệ.

**QUAN TRỌNG**: Node này RESET `sub_queries` và tăng `retry_count`.

**Parameters**:
- `state`: GraphState với fields `reformulated_question`, `generation`, `retry_count`
- `llm`: LLM instance để viết lại câu hỏi

**Returns**:
```python
{
    "sub_queries": List[str],  # Câu hỏi mới đã được viết lại (RESET)
    "retry_count": int  # retry_count + 1
}
```

**Raises**:
- `ValueError`: Nếu `reformulated_question` rỗng

---

## Graph Assembly

### `graph.py`

#### Function: `build_graph(*, llm, vector_store, csv_agent, max_retries: int | None = None) -> Any`

**Mô tả**: Xây dựng LangGraph app hoàn chỉnh cho chatbot.

**Parameters**:
- `llm`: LLM instance dùng cho social/contextualize/generator/rewriter
- `vector_store`: Chroma (hoặc vector store tương thích) cho medical docs
- `csv_agent`: Pandas DataFrame agent (đối tượng có `.invoke()`)
- `max_retries`: Số lần tối đa cho retry loop (nếu `None`, dùng từ `CHATBOT_CONFIG`)

**Returns**: Compiled LangGraph app (có thể gọi `.invoke()` hoặc `.stream()`)

**Graph Structure**:

```
Start → guardrails
    ├─→ route="social" → social_bot → END
    └─→ route="health" → contextualize → query_analysis
            ├─→ intent="realtime_data" → csv_node → generator → answer_grader → END
            ├─→ intent="chunked_data" → retriever_node → doc_grader → generator → answer_grader → END
            ├─→ intent="both" → csv_node → retriever_node → doc_grader → generator → answer_grader → END
            └─→ intent="fall_back" → generator → answer_grader → END
                                    ↓ (if answer_valid=False and retry_count < max_retries)
                                rewriter → query_analysis (retry loop)
```

**Routing Functions**:

1. `route_from_guardrails(state: GraphState) -> str`:
   - Trả về `"social_bot"` nếu `route="social"`
   - Trả về `"contextualize"` nếu `route="health"`

2. `route_from_query_analysis(state: GraphState) -> str`:
   - Trả về `"csv_only"` nếu `intent="realtime_data"`
   - Trả về `"retriever_only"` nếu `intent="chunked_data"`
   - Trả về `"both"` nếu `intent="both"`
   - Trả về `"no_rag"` nếu `intent="fall_back"`

3. `route_after_csv(state: GraphState) -> str`:
   - Trả về `"retriever_node"` nếu `intent="both"`
   - Trả về `"generator"` nếu `intent="realtime_data"`

4. `route_from_answer_grader(state: GraphState) -> str`:
   - Trả về `"end"` nếu `answer_valid=True` hoặc `retry_count >= max_retries`
   - Trả về `"rewrite"` nếu `answer_valid=False` và `retry_count < max_retries`

**Memory**: Sử dụng `MemorySaver` để lưu chat history theo `thread_id`.

**Ví dụ**:
```python
from src.chatbot.graph import build_graph
from src.chatbot.app_runtime import create_chatbot_app

app = create_chatbot_app()  # Sử dụng build_graph bên trong

# Invoke với thread_id
config = {"configurable": {"thread_id": "user1"}}
result = app.invoke(
    {
        "messages": [HumanMessage(content="CVS là gì?")],
        "original_question": "CVS là gì?",
        # ... các fields khác
    },
    config=config
)
```

---

## Runtime và Interface

### `llm_factory.py`

#### Function: `configure_google_client(config: Optional[ChatbotConfig] = None) -> ChatbotConfig`

**Mô tả**: Cấu hình Google GenAI client với API key từ config.

**Parameters**:
- `config`: Optional ChatbotConfig (mặc định: `CHATBOT_CONFIG`)

**Returns**: ChatbotConfig instance

**Raises**:
- `ValueError`: Nếu `GOOGLE_API_KEY` chưa được cấu hình

---

#### Function: `create_production_llm(config: Optional[ChatbotConfig] = None) -> ChatGoogleGenerativeAI`

**Mô tả**: Tạo `ChatGoogleGenerativeAI` instance sử dụng config của project.

**Parameters**:
- `config`: Optional ChatbotConfig (mặc định: `CHATBOT_CONFIG`)

**Returns**: `ChatGoogleGenerativeAI` instance với:
- `model`: Từ `config.llm_model_name` (mặc định: `"gemini-2.5-flash"`)
- `temperature`: 0.2

**Ví dụ**:
```python
from src.chatbot.llm_factory import create_production_llm

llm = create_production_llm()
response = llm.invoke("CVS là gì?")
print(response.content)
```

---

#### Function: `create_production_embeddings(config: Optional[ChatbotConfig] = None) -> GoogleGenerativeAIEmbeddings`

**Mô tả**: Tạo `GoogleGenerativeAIEmbeddings` instance sử dụng config của project.

**Parameters**:
- `config`: Optional ChatbotConfig (mặc định: `CHATBOT_CONFIG`)

**Returns**: `GoogleGenerativeAIEmbeddings` instance với model từ `config.embedding_model_name`

**Lưu ý**: Khuyến nghị dùng `create_huggingface_embeddings()` từ `vector_store.py` để tránh quota limit.

---

### `app_runtime.py`

#### Function: `create_chatbot_app() -> Any`

**Mô tả**: Tạo LangGraph app hoàn chỉnh với production dependencies (LLM thật, vector store thật, CSV agent thật).

**Returns**: Compiled LangGraph app

**Process**:
1. Validate config (`CHATBOT_CONFIG.validate()`)
2. Tạo LLM (`create_production_llm()`)
3. Build/load vector store (`build_or_load_medical_vector_store()`)
4. Load CSV và tạo agent (`load_summary_dataframe()`, `create_summary_agent()`)
5. Build graph (`build_graph()`)

**Ví dụ**:
```python
from src.chatbot.app_runtime import create_chatbot_app

app = create_chatbot_app()
# App đã sẵn sàng để sử dụng
```

---

### `chat_interface.py`

#### Function: `chat_interface(user_input: str, thread_id: str = "local_user_1", *, app: Optional[Any] = None) -> str`

**Mô tả**: Interface công khai để gửi message đến chatbot và nhận câu trả lời.

**Parameters**:
- `user_input`: `str` - Câu hỏi / tin nhắn của người dùng
- `thread_id`: `str` - ID phiên chat (dùng cho MemorySaver, mặc định: `"local_user_1"`)
- `app`: `Optional[Any]` - Optional compiled LangGraph app (nếu `None`, gọi `create_chatbot_app()`)

**Returns**: `str` - Câu trả lời cuối cùng của chatbot

**Error Handling**:
- Nếu có exception, trả về message lỗi: `"Hệ thống đang gặp sự cố kỹ thuật."`

**Ví dụ**:
```python
from src.chatbot.chat_interface import chat_interface

# Sử dụng app mặc định (tự động tạo)
answer = chat_interface("CVS là gì?", thread_id="user1")
print(answer)

# Sử dụng app custom (cho testing)
from src.chatbot.app_runtime import create_chatbot_app
app = create_chatbot_app()
answer = chat_interface("Hi", thread_id="user1", app=app)
```

**Lưu ý**: 
- `thread_id` giúp ghi nhớ lịch sử hội thoại cho từng user/phiên chat
- Có thể map `thread_id` = ID user trong hệ thống login của app

---

### `app.py`

#### Function: `main() -> None`

**Mô tả**: CLI application để chat với chatbot qua console.

**Usage**:
```powershell
cd D:\AI_Final\Final_AI_Project
.\.venv_chatbot\Scripts\Activate.ps1
python src/chatbot/app.py
```

**Features**:
- Sử dụng `thread_id="cli_user"` cố định
- Gõ `exit` hoặc để trống rồi Enter để thoát
- Hỗ trợ `Ctrl+C` để thoát

---

## Ví dụ sử dụng

### Ví dụ 1: Sử dụng chat_interface (đơn giản nhất)

```python
from src.chatbot.chat_interface import chat_interface

# Câu hỏi đơn giản
answer = chat_interface("CVS là gì?", thread_id="user1")
print(answer)

# Câu hỏi về thống kê
answer = chat_interface(
    "Trung bình thời lượng các phiên đo gần đây là bao nhiêu phút?",
    thread_id="user1"
)
print(answer)

# Câu hỏi kết hợp
answer = chat_interface(
    "Thống kê thời lượng các session và giải thích CVS là gì.",
    thread_id="user1"
)
print(answer)
```

---

### Ví dụ 2: Sử dụng app trực tiếp (nâng cao)

```python
from src.chatbot.app_runtime import create_chatbot_app
from langchain_core.messages import HumanMessage

# Tạo app
app = create_chatbot_app()

# Config với thread_id
config = {"configurable": {"thread_id": "user1"}}

# Initial state
inputs = {
    "messages": [HumanMessage(content="CVS là gì?")],
    "original_question": "CVS là gì?",
    "reformulated_question": "",
    "generation": "",
    "analyzed_intent": "fall_back",
    "sub_queries": [],
    "context": [],
    "retry_count": 0,
    "answer_valid": True,
}

# Invoke
result = app.invoke(inputs, config=config)
print(result["generation"])
```

---

### Ví dụ 3: Stream execution để trace

```python
from src.chatbot.app_runtime import create_chatbot_app
from langchain_core.messages import HumanMessage

app = create_chatbot_app()
config = {"configurable": {"thread_id": "user1"}}

inputs = {
    "messages": [HumanMessage(content="CVS là gì?")],
    "original_question": "CVS là gì?",
    "reformulated_question": "",
    "generation": "",
    "analyzed_intent": "fall_back",
    "sub_queries": [],
    "context": [],
    "retry_count": 0,
    "answer_valid": True,
}

# Stream để xem từng node được thực thi
for event in app.stream(inputs, config=config, stream_mode="values"):
    for node_name, node_output in event.items():
        if node_name != "__end__":
            print(f"Node '{node_name}' executed")
            if "generation" in node_output:
                print(f"  Generation: {node_output['generation'][:100]}...")
```

---

### Ví dụ 4: Tích hợp vào GUI/Desktop App

```python
from src.chatbot.chat_interface import chat_interface

class ChatbotWidget:
    def __init__(self):
        self.thread_id = "gui_user_1"
    
    def ask(self, question: str) -> str:
        """Gửi câu hỏi đến chatbot và nhận câu trả lời."""
        return chat_interface(question, thread_id=self.thread_id)

# Sử dụng trong GUI
widget = ChatbotWidget()
answer = widget.ask("Tôi bị mỏi mắt khi làm việc với máy tính, tôi nên làm gì?")
print(answer)
```

---

## Testing

### Chạy tất cả tests

```powershell
cd D:\AI_Final\Final_AI_Project
.\.venv_chatbot\Scripts\Activate.ps1
pytest src/chatbot/test/ -v -s
```

### Chạy test cụ thể

```powershell
# Test config
pytest src/chatbot/test/test_config.py -v -s

# Test tools (vector_store, csv_loader)
pytest src/chatbot/test/test_tools.py -v -s

# Test từng node
pytest src/chatbot/test/test_nodes.py -v -s

# Test toàn bộ graph
pytest src/chatbot/test/test_graph.py -v -s

# Test chat interface
pytest src/chatbot/test/test_chat_interface.py -v -s

# Test fixes (contextualize, context reset, HuggingFace embeddings)
pytest src/chatbot/test/test_fixes.py -v -s
```

### Test HuggingFace embeddings với ChromaDB

```powershell
python src/chatbot/test/test_chromadb_huggingface.py
```

### Test Google API Key

```powershell
python -m src.chatbot.test.test_google_api_key
```

---

## Troubleshooting

### Lỗi: `GOOGLE_API_KEY is not set`

**Nguyên nhân**: File `.env` chưa được tạo hoặc không có `GOOGLE_API_KEY`.

**Giải pháp**:
1. Tạo file `src/chatbot/.env` hoặc `.env` ở project root
2. Thêm dòng: `GOOGLE_API_KEY="your_api_key_here"`

---

### Lỗi: `ModuleNotFoundError: No module named 'cv2'`

**Nguyên nhân**: Đang dùng Python từ venv sai (ví dụ: dùng `.venv_chatbot` thay vì `src/.venv`).

**Giải pháp**:
- Khi chạy `src/main.py`: activate `src/.venv`
- Khi chạy chatbot: activate `.venv_chatbot`

---

### Lỗi: `Collection expecting embedding with dimension of 64, got 384`

**Nguyên nhân**: ChromaDB cũ được tạo với embedding dimension khác (ví dụ: Google embeddings dimension 768, HuggingFace dimension 384).

**Giải pháp**:
- Code tự động detect và rebuild ChromaDB khi có dimension mismatch
- Hoặc xóa thư mục `data/chroma_db` và rebuild:
```python
from src.chatbot.tools.vector_store import build_or_load_medical_vector_store
from src.chatbot.config import CHATBOT_CONFIG

vs = build_or_load_medical_vector_store(CHATBOT_CONFIG, force_rebuild=True)
```

---

### Lỗi: `ResourceExhausted: 429 You exceeded your current quota`

**Nguyên nhân**: Google API quota đã hết (Free Tier có giới hạn).

**Giải pháp**:
- Dùng HuggingFace embeddings (local, miễn phí, không có quota limit):
```python
from src.chatbot.tools.vector_store import build_or_load_medical_vector_store
from src.chatbot.config import CHATBOT_CONFIG

vs = build_or_load_medical_vector_store(
    CHATBOT_CONFIG,
    use_huggingface=True,  # Dùng HuggingFace embeddings
    force_rebuild=True
)
```

---

## Tóm tắt các điểm quan trọng

1. **Context Reset**: Tất cả các node RESET context fields (không append) để tránh tích lũy qua nhiều lượt hỏi.

2. **Contextualize**: Chỉ viết lại câu hỏi mới nhất, không tổng hợp các câu hỏi trước.

3. **Embeddings**: Mặc định dùng HuggingFace embeddings (local, miễn phí) thay vì Google embeddings (có quota limit).

4. **Memory**: Sử dụng `thread_id` để ghi nhớ lịch sử hội thoại cho từng user/phiên chat.

5. **Error Handling**: Code có xử lý lỗi robust cho API quota, parsing errors, và dimension mismatch.

---

## Tham khảo

- LangGraph Documentation: https://langchain-ai.github.io/langgraph/
- LangChain Documentation: https://python.langchain.com/
- ChromaDB Documentation: https://docs.trychroma.com/
- Google Gemini API: https://ai.google.dev/

