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
# Dùng langchain_community.vectorstores.Chroma để tương thích với LangChain 0.3.x
# langchain-chroma yêu cầu langchain-core>=1.0.0 (không tương thích với 0.3.x)
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
    
    Hàm này trả về các đường dẫn mặc định cho:
    - Thư mục chứa medical documents (docs_dir)
    - Thư mục lưu trữ Chroma vector store (persist_dir)

    Args
    ----
    config:
        Chatbot configuration instance chứa chroma_persist_directory.

    Returns
    -------
    MedicalVectorStorePaths
        Paths cho documents và Chroma DB persistence.
    """
    # Dữ liệu medical_docs đặt trong src/data/medical_docs theo cấu trúc project hiện tại
    docs_dir = PROJECT_ROOT / "src" / "data" / "medical_docs"
    
    # Trả về MedicalVectorStorePaths với docs_dir và persist_dir từ config
    return MedicalVectorStorePaths(
        docs_dir=docs_dir,
        persist_dir=config.chroma_persist_directory,
    )


def load_medical_documents(docs_dir: Path) -> List[Document]:
    """Load medical documents from a directory as LangChain ``Document`` objects.
    
    Hàm này load tất cả các file .txt trong thư mục docs_dir (bao gồm cả subdirectories)
    và chuyển đổi thành LangChain Document objects để dùng cho vector store.

    Parameters
    ----------
    docs_dir:
        Directory chứa medical documents. Hiện tại chỉ hỗ trợ file ``.txt``.

    Returns
    -------
    list of Document
        Danh sách documents đã được load.

    Raises
    ------
    FileNotFoundError
        Nếu ``docs_dir`` không tồn tại.
    ValueError
        Nếu không tìm thấy file .txt nào trong thư mục.
    """
    # Kiểm tra thư mục có tồn tại không
    if not docs_dir.exists():
        raise FileNotFoundError(f"Medical docs directory not found: {docs_dir}")

    # Tạo DirectoryLoader để load tất cả file .txt (bao gồm cả subdirectories)
    loader = DirectoryLoader(
        str(docs_dir),
        glob="**/*.txt",  # Tìm tất cả file .txt trong mọi subdirectory
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True},  # Tự động detect encoding để tránh UnicodeDecodeError
        show_progress=True,  # Hiển thị progress bar khi load
        use_multithreading=True,  # Dùng multithreading để load nhanh hơn
    )
    
    # Load tất cả documents
    docs = loader.load()
    
    # Validate: kiểm tra có documents không
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
        """Tạo embedding cục bộ đơn giản, deterministic dựa trên hash.
        
        Hàm này chỉ dùng như fallback khi Google API hết quota.
        Embedding này KHÔNG chính xác về mặt semantic, chỉ để tránh crash.
        
        Parameters
        ----------
        text:
            Text cần embed.
        dim:
            Số chiều của embedding vector (mặc định 64).
        
        Returns
        -------
        List[float]
            Embedding vector (deterministic, nhưng không semantic).
        """
        # Tính hash của text (deterministic: cùng text -> cùng hash)
        h = abs(hash(text))
        vec = []
        
        # Sinh vector từ hash: mỗi bit của hash được chuyển thành giá trị float [0, 1]
        for i in range(dim):
            # Sinh pseudo-random từ hash, nhưng deterministic
            # Shift hash và mask để lấy 16 bits, normalize về [0, 1]
            v = ((h >> (i % 32)) & 0xFFFF) / 65535.0
            vec.append(float(v))
        
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:  # type: ignore[override]
        """Embed danh sách documents bằng Google API hoặc fallback local.
        
        Nếu Google API hoạt động: dùng Google embeddings (chính xác, semantic).
        Nếu Google API lỗi (quota, network): fallback sang local hash-based embeddings.
        
        Parameters
        ----------
        texts:
            Danh sách texts cần embed.
        
        Returns
        -------
        List[List[float]]
            Danh sách embedding vectors (mỗi text một vector).
        """
        # Thử dùng Google API nếu còn enabled
        if self._remote_enabled:
            try:
                return self._remote.embed_documents(texts)
            except Exception as exc:  # pragma: no cover - fallback path
                # Nếu lỗi (quota, network, etc.): in warning và disable remote
                print(
                    "Warning: Google embeddings failed, falling back to local embeddings.",
                    "Error:",
                    repr(exc),
                )
                self._remote_enabled = False
        
        # Fallback: dùng local hash-based embeddings (không chính xác, nhưng tránh crash)
        return [self._local_embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:  # type: ignore[override]
        """Embed một query text bằng Google API hoặc fallback local.
        
        Tương tự embed_documents, nhưng dành cho single query text.
        
        Parameters
        ----------
        text:
            Query text cần embed.
        
        Returns
        -------
        List[float]
            Embedding vector cho query.
        """
        # Thử dùng Google API nếu còn enabled
        if self._remote_enabled:
            try:
                return self._remote.embed_query(text)
            except Exception as exc:  # pragma: no cover - fallback path
                # Nếu lỗi: in warning và disable remote
                print(
                    "Warning: Google query embedding failed, falling back to local embedding.",
                    "Error:",
                    repr(exc),
                )
                self._remote_enabled = False
        
        # Fallback: dùng local hash-based embedding
        return self._local_embed(text)


def create_huggingface_embeddings() -> Embeddings:
    """Create HuggingFace embeddings instance (local, miễn phí, không có quota limit).
    
    Hàm này tạo embeddings model từ HuggingFace, chạy hoàn toàn local trên CPU.
    Ưu điểm: không cần API key, không có quota limit, miễn phí hoàn toàn.
    Model: sentence-transformers/all-MiniLM-L6-v2 (nhẹ, nhanh, hỗ trợ đa ngôn ngữ).
    
    Returns
    -------
    Embeddings
        LangChain ``Embeddings`` implementation using HuggingFace sentence-transformers.
    
    Note
    ----
    Model sẽ được download tự động lần đầu tiên (khoảng 80MB).
    Sau đó sẽ cache local, không cần download lại.
    """
    # Thử dùng langchain-huggingface trước (nếu có)
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},  # Chạy trên CPU (có thể đổi 'cuda' nếu có GPU)
            encode_kwargs={'normalize_embeddings': True}  # Normalize để tối ưu similarity search
        )
    except ImportError:
        # Fallback: dùng HuggingFaceEmbeddings từ langchain-community
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except ImportError:
            raise ImportError(
                "HuggingFace embeddings không khả dụng. "
                "Hãy cài đặt: pip install langchain-huggingface hoặc langchain-community"
            )


def create_google_embeddings(config: ChatbotConfig = CHATBOT_CONFIG) -> Embeddings:
    """Create Google Generative AI embeddings instance using project config.
    
    Hàm này tạo embeddings từ Google API (có quota limit).
    Nếu hết quota, sẽ fallback sang local hash-based embeddings (không chính xác).
    
    Parameters
    ----------
    config:
        Chatbot configuration containing API key and embedding model name.

    Returns
    -------
    Embeddings
        LangChain ``Embeddings`` implementation using Google GenAI với fallback.

    Raises
    ------
    ValueError
        If ``GOOGLE_API_KEY`` is missing.
    
    Note
    ----
    Khuyến nghị: Dùng ``create_huggingface_embeddings()`` thay vì hàm này để tránh quota limit.
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
    Hàm này tạo Chroma vector store từ documents và embeddings, có thể persist hoặc in-memory.

    Parameters
    ----------
    documents:
        Iterable of LangChain ``Document`` objects cần index vào vector store.
    embeddings:
        Embeddings implementation để tạo embeddings cho documents.
    persist_directory:
        Optional directory để persist Chroma DB. Nếu ``None``, DB là in-memory (không lưu).

    Returns
    -------
    Chroma
        Chroma vector store đã được xây dựng, sẵn sàng cho truy vấn.
    
    Raises
    ------
    ValueError
        Nếu documents rỗng.
    """
    # Chuyển iterable thành list để validate và dùng nhiều lần
    doc_list = list(documents)
    
    # Validate: kiểm tra có documents không
    if not doc_list:
        raise ValueError("No documents provided to build Chroma store.")

    # Chuyển Path thành string (Chroma yêu cầu string path)
    persist_str = str(persist_directory) if persist_directory is not None else None
    
    # Tạo Chroma vector store từ documents và embeddings
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
    use_huggingface: bool = True,
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
        ``create_huggingface_embeddings()`` (mặc định) hoặc ``create_google_embeddings(config)``.
    force_rebuild:
        Nếu ``True`` luôn rebuild DB từ tài liệu.
    use_huggingface:
        Nếu ``True`` (mặc định), dùng HuggingFace embeddings (local, miễn phí).
        Nếu ``False``, dùng Google embeddings (có quota limit).

    Returns
    -------
    Chroma
        Vector store đã sẵn sàng cho truy vấn.
    """
    # Lấy paths (mặc định hoặc custom)
    paths = paths or get_default_paths(config)
    persist_dir = paths.persist_dir
    
    # Mặc định dùng HuggingFace embeddings (local, miễn phí, không có quota limit)
    if embeddings is None:
        if use_huggingface:
            embeddings = create_huggingface_embeddings()
        else:
            embeddings = create_google_embeddings(config)

    # Logic: rebuild nếu force_rebuild=True hoặc persist_dir chưa tồn tại
    if force_rebuild or not persist_dir.exists():
        # Load documents từ docs_dir và build vector store mới
        docs = load_medical_documents(paths.docs_dir)
        vs = build_chroma_from_documents(docs, embeddings, persist_directory=persist_dir)
    else:
        # Load vector store đã tồn tại từ persist_dir
        # Nếu có lỗi (do version mismatch, dimension mismatch, corrupted), force rebuild
        try:
            vs = Chroma(
                persist_directory=str(persist_dir),
                embedding_function=embeddings,
            )
            # Test query để đảm bảo database hoạt động và embedding dimension khớp
            test_query = "test"
            _ = vs.as_retriever(search_kwargs={"k": 1}).invoke(test_query)
        except (KeyError, ValueError, Exception) as e:
            # Nếu có lỗi load database (version mismatch, dimension mismatch, corrupted, etc.), rebuild
            error_msg = str(e)
            if "dimension" in error_msg.lower() or "_type" in error_msg.lower():
                print(f"Warning: ChromaDB dimension/version mismatch ({error_msg[:100]}), rebuilding...")
            else:
                print(f"Warning: Failed to load existing ChromaDB ({error_msg[:100]}), rebuilding...")
            # Force rebuild: xóa database cũ và tạo mới
            import shutil
            if persist_dir.exists():
                shutil.rmtree(persist_dir)
            docs = load_medical_documents(paths.docs_dir)
            vs = build_chroma_from_documents(docs, embeddings, persist_directory=persist_dir)
    
    return vs


