"""Vector store utilities for medical knowledge documents.

Module này xây dựng Chroma vector DB dùng embedding miễn phí của Google
(``models/embedding-001``) cho các tài liệu y tế dạng text.

Thiết kế:
- Core logic tách ra thành hàm thuần (nhận sẵn embeddings, danh sách document).
- Hàm helper riêng để tạo Google embeddings từ ``ChatbotConfig``.
=> Dễ kiểm thử (có thể dùng DummyEmbeddings trong pytest, không gọi API thật).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import google.generativeai as genai
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.chatbot.config import CHATBOT_CONFIG, ChatbotConfig


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class MedicalVectorStorePaths:
    """Group of filesystem paths used by the medical vector store."""

    docs_dir: Path
    persist_dir: Path


def get_default_paths(config: ChatbotConfig = CHATBOT_CONFIG) -> MedicalVectorStorePaths:
    """Return default paths for medical docs and Chroma persistence.

    Args
    ----
    config:
        Chatbot configuration instance.

    Returns
    -------
    MedicalVectorStorePaths
        Paths for documents and Chroma DB.
    """
    # Dữ liệu medical_docs đặt trong src/data/medical_docs theo cấu trúc project hiện tại.
    docs_dir = PROJECT_ROOT / "src" / "data" / "medical_docs"
    return MedicalVectorStorePaths(
        docs_dir=docs_dir,
        persist_dir=config.chroma_persist_directory,
    )


def load_medical_documents(docs_dir: Path) -> List[Document]:
    """Load medical documents from a directory as LangChain ``Document`` objects.

    Parameters
    ----------
    docs_dir:
        Directory containing medical documents. Currently expects ``.txt`` files.

    Returns
    -------
    list of Document
        Loaded documents.

    Raises
    ------
    FileNotFoundError
        If ``docs_dir`` does not exist.
    ValueError
        If no documents are found in the directory.
    """
    if not docs_dir.exists():
        raise FileNotFoundError(f"Medical docs directory not found: {docs_dir}")

    loader = DirectoryLoader(
        str(docs_dir),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True},
        show_progress=True,
        use_multithreading=True,
    )
    docs = loader.load()
    if not docs:
        raise ValueError(f"No .txt medical documents found in {docs_dir}")
    return docs


class SafeGoogleEmbeddings(Embeddings):
    """Wrapper quanh GoogleGenerativeAIEmbeddings với fallback khi hết quota.

    Nếu gọi API embedding bị lỗi (ví dụ 429 ResourceExhausted), sẽ fallback sang
    một embedding cục bộ đơn giản (deterministic) để tránh làm hỏng toàn bộ pipeline.
    """

    def __init__(self, model_name: str, api_key: str) -> None:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        genai.configure(api_key=api_key)
        self._remote = GoogleGenerativeAIEmbeddings(model=model_name)
        self._remote_enabled = True

    def _local_embed(self, text: str, dim: int = 64) -> List[float]:
        # Embedding cục bộ rất đơn giản, deterministic dựa trên hash,
        # chỉ dùng như fallback khi hết quota.
        h = abs(hash(text))
        vec = []
        for i in range(dim):
            # Sinh pseudo-random từ hash, nhưng deterministic.
            v = ((h >> (i % 32)) & 0xFFFF) / 65535.0
            vec.append(float(v))
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:  # type: ignore[override]
        if self._remote_enabled:
            try:
                return self._remote.embed_documents(texts)
            except Exception as exc:  # pragma: no cover - fallback path
                print(
                    "Warning: Google embeddings failed, falling back to local embeddings.",
                    "Error:",
                    repr(exc),
                )
                self._remote_enabled = False
        return [self._local_embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:  # type: ignore[override]
        if self._remote_enabled:
            try:
                return self._remote.embed_query(text)
            except Exception as exc:  # pragma: no cover - fallback path
                print(
                    "Warning: Google query embedding failed, falling back to local embedding.",
                    "Error:",
                    repr(exc),
                )
                self._remote_enabled = False
        return self._local_embed(text)


def create_google_embeddings(config: ChatbotConfig = CHATBOT_CONFIG) -> Embeddings:
    """Create Google Generative AI embeddings instance using project config.

    Parameters
    ----------
    config:
        Chatbot configuration containing API key and embedding model name.

    Returns
    -------
    Embeddings
        LangChain ``Embeddings`` implementation using Google GenAI.

    Raises
    ------
    ValueError
        If ``GOOGLE_API_KEY`` is missing.
    """
    if not config.google_api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not configured. "
            "Hãy đảm bảo đã thêm vào file .env của chatbot."
        )

    # Trả về wrapper an toàn, nội bộ sẽ dùng GoogleGenerativeAIEmbeddings
    # và fallback cục bộ nếu bị lỗi quota hoặc mạng.
    return SafeGoogleEmbeddings(
        model_name=config.embedding_model_name,
        api_key=config.google_api_key,
    )


def build_chroma_from_documents(
    documents: Iterable[Document],
    embeddings: Embeddings,
    persist_directory: Optional[Path] = None,
) -> Chroma:
    """Build a Chroma vector store from an iterable of documents.

    Đây là hàm core, thuần (không phụ thuộc config), thuận tiện cho pytest.

    Parameters
    ----------
    documents:
        Iterable of LangChain ``Document`` objects.
    embeddings:
        Embeddings implementation to use.
    persist_directory:
        Optional directory to persist Chroma DB. If ``None``, DB is in-memory.

    Returns
    -------
    Chroma
        Constructed Chroma vector store.
    """
    doc_list = list(documents)
    if not doc_list:
        raise ValueError("No documents provided to build Chroma store.")

    persist_str = str(persist_directory) if persist_directory is not None else None
    return Chroma.from_documents(
        documents=doc_list,
        embedding=embeddings,
        persist_directory=persist_str,
    )


def build_or_load_medical_vector_store(
    config: ChatbotConfig = CHATBOT_CONFIG,
    *,
    paths: Optional[MedicalVectorStorePaths] = None,
    embeddings: Optional[Embeddings] = None,
    force_rebuild: bool = False,
) -> Chroma:
    """Build or load the medical Chroma vector store.

    High-level convenience function dùng cho ứng dụng thật.

    Logic:
    - Nếu ``force_rebuild=True`` hoặc folder persist chưa tồn tại:
        -> load tài liệu từ ``docs_dir`` và xây mới vector store.
    - Ngược lại:
        -> load Chroma từ ``persist_dir``.

    Parameters
    ----------
    config:
        Chatbot configuration.
    paths:
        Optional custom paths; nếu ``None`` dùng ``get_default_paths``.
    embeddings:
        Optional embeddings implementation. Nếu ``None`` sẽ tạo từ
        ``create_google_embeddings(config)``.
    force_rebuild:
        Nếu ``True`` luôn rebuild DB từ tài liệu.

    Returns
    -------
    Chroma
        Vector store đã sẵn sàng cho truy vấn.
    """
    paths = paths or get_default_paths(config)
    persist_dir = paths.persist_dir
    embeddings = embeddings or create_google_embeddings(config)

    if force_rebuild or not persist_dir.exists():
        docs = load_medical_documents(paths.docs_dir)
        vs = build_chroma_from_documents(docs, embeddings, persist_directory=persist_dir)
    else:
        vs = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
        )
    return vs


