"""
Basic tests for ``src.chatbot.config`` module.

Các test này dùng cho môn học, giúp kiểm tra nhanh:
- Đọc biến môi trường vào ChatbotConfig.from_env().
- Giá trị mặc định khi thiếu env.
- Hàm validate() bắt lỗi khi thiếu GOOGLE_API_KEY.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Đảm bảo project root (chứa thư mục src/) nằm trong sys.path
# File path: <project>/src/chatbot/test/test_config.py -> parents[4] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chatbot.config import ChatbotConfig


def _clear_env(keys: list[str]) -> None:
    """Helper: remove specific keys from os.environ if present."""
    for k in keys:
        os.environ.pop(k, None)


def test_from_env_uses_defaults_when_missing(monkeypatch) -> None:
    """Config.from_env should fall back to sane defaults."""
    keys = [
        "GOOGLE_API_KEY",
        "LANGCHAIN_API_KEY",
        "LANGCHAIN_PROJECT",
        "LANGCHAIN_ENDPOINT",
        "LANGCHAIN_TRACING_V2",
        "LLM_MODEL_NAME",
        "EMBEDDING_MODEL_NAME",
        "CHROMA_PERSIST_DIRECTORY",
        "CSV_FILE_PATH",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "K_RETRIEVAL",
        "MAX_RETRIES",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)

    cfg = ChatbotConfig.from_env()

    # Defaults theo design docs
    assert cfg.llm_model_name == "gemini-2.5-flash"
    assert cfg.embedding_model_name == "models/embedding-001"
    assert cfg.chunk_size == 1000
    assert cfg.chunk_overlap == 200
    assert cfg.k_retrieval == 3
    assert cfg.max_retries == 3

    # Đường dẫn tương đối tới thư mục dự án
    assert isinstance(cfg.chroma_persist_directory, Path)
    assert isinstance(cfg.csv_file_path, Path)


def test_from_env_reads_overrides(monkeypatch) -> None:
    """Config.from_env should respect overrides from environment."""
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy-key")
    monkeypatch.setenv("LLM_MODEL_NAME", "gemini-2.5-flash")
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "models/custom-embedding")
    monkeypatch.setenv("CHUNK_SIZE", "512")
    monkeypatch.setenv("CHUNK_OVERLAP", "64")
    monkeypatch.setenv("K_RETRIEVAL", "5")
    monkeypatch.setenv("MAX_RETRIES", "7")

    cfg = ChatbotConfig.from_env()

    assert cfg.google_api_key == "dummy-key"
    assert cfg.llm_model_name == "gemini-2.5-flash"
    assert cfg.embedding_model_name == "models/custom-embedding"
    assert cfg.chunk_size == 512
    assert cfg.chunk_overlap == 64
    assert cfg.k_retrieval == 5
    assert cfg.max_retries == 7


def test_validate_requires_google_api_key(monkeypatch) -> None:
    """validate() should raise when GOOGLE_API_KEY is missing."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    cfg = ChatbotConfig.from_env()

    raised = False
    try:
        cfg.validate()
    except ValueError as exc:
        raised = True
        print("Input GOOGLE_API_KEY:", str(cfg.google_api_key).encode("utf-8", errors="ignore"))
        print("Output error message:", str(exc).encode("utf-8", errors="ignore"))

    assert raised, "Expected ValueError when GOOGLE_API_KEY is missing."


def test_validate_passes_with_valid_params(monkeypatch) -> None:
    """Happy path: validate() should not raise with correct settings."""
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy-key")
    monkeypatch.setenv("CHUNK_SIZE", "1000")
    monkeypatch.setenv("CHUNK_OVERLAP", "200")
    monkeypatch.setenv("K_RETRIEVAL", "3")
    monkeypatch.setenv("MAX_RETRIES", "2")

    cfg = ChatbotConfig.from_env()

    # In ra để bạn dễ kiểm tra khi chạy pytest -s
    print("Config validation input:")
    print("  chunk_size:", cfg.chunk_size)
    print("  chunk_overlap:", cfg.chunk_overlap)
    print("  k_retrieval:", cfg.k_retrieval)
    print("  max_retries:", cfg.max_retries)

    cfg.validate()  # Không được raise


