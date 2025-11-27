"""
Script: build_vector_store.py

Mục đích:
- Sử dụng embedding thật của Google (embedding-001) với GOOGLE_API_KEY từ .env.
- Đọc toàn bộ dữ liệu y tế trong `data/medical_docs/*.txt`.
- Xây dựng lại Chroma vector store thật và lưu vào `data/chroma_db`.

Cách chạy (PowerShell):

    cd D:\AI_Final\Final_AI_Project
    .\.venv_chatbot\Scripts\Activate.ps1
    python -m src.chatbot.build_vector_store

Yêu cầu:
- Đã cấu hình GOOGLE_API_KEY trong `src/chatbot/.env` hoặc `.env` ở project root.
- Đã cài các thư viện trong `src/chatbot/requirements.txt`.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chatbot.config import CHATBOT_CONFIG  # noqa: E402
from src.chatbot.tools.vector_store import (  # noqa: E402
    build_or_load_medical_vector_store,
    get_default_paths,
)


def main() -> None:
    print("=== Build REAL Medical Vector Store (Chroma + Google Embeddings) ===")

    # Kiểm tra cấu hình (bao gồm GOOGLE_API_KEY)
    try:
        CHATBOT_CONFIG.validate()
    except Exception as exc:
        print("Config validation FAILED:")
        print("  Error:", repr(exc))
        print(
            "\nHãy kiểm tra lại file .env (src/chatbot/.env hoặc .env ở root) "
            "và chắc chắn đã đặt GOOGLE_API_KEY=\"...\"."
        )
        sys.exit(1)

    paths = get_default_paths(CHATBOT_CONFIG)
    print("Docs directory :", paths.docs_dir)
    print("Chroma DB path :", paths.persist_dir)

    try:
        # force_rebuild=True để chắc chắn đọc lại toàn bộ dữ liệu thật
        _ = build_or_load_medical_vector_store(
            config=CHATBOT_CONFIG,
            paths=paths,
            embeddings=None,  # None => dùng Google embeddings thật
            force_rebuild=True,
        )
        print("\nVector store đã được xây dựng thành công với dữ liệu thật.")
        print("Bạn có thể kiểm tra thư mục:", paths.persist_dir)
    except Exception as exc:  # pragma: no cover - script thủ công
        print("\nXây dựng vector store THẤT BẠI.")
        print("Error type  :", type(exc))
        print("Error detail:", repr(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()


