"""
Chatbot configuration module.

This module centralizes all configuration for the RAG healthcare chatbot,
including model names, data paths, and API credentials.

The goal is:
- Không hard-code API key trong code.
- Dễ dàng thay đổi model embedding/LLM miễn phí của Google.
- Tách biệt cấu hình chatbot khỏi phần còn lại của hệ thống vision.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


# Tải biến môi trường từ file .env.
# Ưu tiên:
#   1) .env ngay trong thư mục src/chatbot (theo yêu cầu của bạn).
#   2) .env ở project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CHATBOT_DIR = Path(__file__).resolve().parent

_LOCAL_ENV = _CHATBOT_DIR / ".env"
_ROOT_ENV = _PROJECT_ROOT / ".env"

if _LOCAL_ENV.exists():
    load_dotenv(dotenv_path=_LOCAL_ENV)
elif _ROOT_ENV.exists():
    load_dotenv(dotenv_path=_ROOT_ENV)
else:
    # Fallback: vẫn gọi load_dotenv() để hỗ trợ trường hợp người dùng
    # đặt .env ở current working directory.
    load_dotenv()


@dataclass(frozen=True)
class ChatbotConfig:
    """Immutable configuration for the healthcare RAG chatbot.

    Tất cả tham số quan trọng đều gom về đây để:
    - Dễ chỉnh sửa cho môn học.
    - Dễ kiểm thử.
    - Tránh rải rác biến môi trường trong nhiều module.

    Notes
    -----
    - API key được đọc từ biến môi trường, KHÔNG ghi cứng trong repo.
    - Sử dụng embedding miễn phí của Google: ``models/embedding-001``.
    - Mặc định dùng model LLM ``gemini-2.5-flash`` (phiên bản mới hơn, nhanh và tiết kiệm).
    """

    # --- API Keys & External Services ---
    google_api_key: Optional[str]
    langsmith_api_key: Optional[str]
    langsmith_project: Optional[str]
    langsmith_endpoint: Optional[str]
    langsmith_tracing: bool

    # --- Model names (Google free tier where possible) ---
    llm_model_name: str
    embedding_model_name: str

    # --- Data & RAG parameters ---
    chroma_persist_directory: Path
    csv_file_path: Path
    chunk_size: int
    chunk_overlap: int
    k_retrieval: int
    max_retries: int

    @staticmethod
    def from_env() -> "ChatbotConfig":
        """Create a :class:`ChatbotConfig` from environment variables.

        Environment variables
        ---------------------
        - ``GOOGLE_API_KEY``: API key cho Google Gemini.
        - ``LANGCHAIN_API_KEY``: API key cho LangSmith (tùy chọn).
        - ``LANGCHAIN_PROJECT``: Tên project LangSmith (tùy chọn).
        - ``LANGCHAIN_ENDPOINT``: Endpoint LangSmith (tùy chọn).
        - ``LANGCHAIN_TRACING_V2``: ``\"true\"/\"false\"`` bật tắt tracing.

        Returns
        -------
        ChatbotConfig
            Đối tượng cấu hình đã đọc toàn bộ thông tin cần thiết.
        """
        project_root = _PROJECT_ROOT
        data_dir = project_root / "src" / "data"

        google_api_key = os.getenv("GOOGLE_API_KEY")

        return ChatbotConfig(
            # API & Observability
            google_api_key=google_api_key,
            langsmith_api_key=os.getenv("LANGCHAIN_API_KEY"),
            langsmith_project=os.getenv("LANGCHAIN_PROJECT"),
            langsmith_endpoint=os.getenv("LANGCHAIN_ENDPOINT"),
            langsmith_tracing=os.getenv("LANGCHAIN_TRACING_V2", "false").lower()
            in {"1", "true", "yes", "y"},
            # Models (free / recommended defaults)
            llm_model_name=os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash"),
            embedding_model_name=os.getenv(
                "EMBEDDING_MODEL_NAME", "models/embedding-001"
            ),
            # Data & RAG parameters
            chroma_persist_directory=Path(
                os.getenv(
                    "CHROMA_PERSIST_DIRECTORY",
                    str(project_root / "data" / "chroma_db"),
                )
            ),
            csv_file_path=Path(
                os.getenv(
                    "CSV_FILE_PATH",
                    str(data_dir / "logs" / "user_health_log.csv"),
                )
            ),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            k_retrieval=int(os.getenv("K_RETRIEVAL", "3")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
        )

    def validate(self) -> None:
        """Validate configuration values and raise explicit errors if invalid.

        Raises
        ------
        ValueError
            Nếu thiếu ``GOOGLE_API_KEY`` hoặc các tham số số học không hợp lệ.
        """
        if not self.google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY is not set. "
                "Vui lòng tạo file .env trong thư mục 'src/chatbot' "
                "hoặc ở project root và thêm GOOGLE_API_KEY=\"...\"."
            )

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative.")
        if self.k_retrieval <= 0:
            raise ValueError("k_retrieval must be positive.")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative.")

    @property
    def has_langsmith(self) -> bool:
        """Check whether LangSmith is fully configured."""
        return bool(self.langsmith_api_key and self.langsmith_project)


# Instance tiện dụng cho các module khác import dùng ngay:
# from src.chatbot.config import CHATBOT_CONFIG
CHATBOT_CONFIG = ChatbotConfig.from_env()


