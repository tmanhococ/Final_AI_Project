r"""
Simple CLI app to chat with the healthcare chatbot via console.

Sử dụng:
    cd D:\AI_Final\Final_AI_Project
    .\.venv_chatbot\Scripts\Activate.ps1
    python -m src.chatbot.app

Chatbot dùng `thread_id = "cli_user"` để ghi nhớ lịch sử hội thoại.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Khi chạy trực tiếp bằng đường dẫn (python src/chatbot/app.py),
# cần đảm bảo project root nằm trong sys.path để import được `src.chatbot.*`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chatbot.chat_interface import chat_interface


def main() -> None:
    print("=== Health Care Chatbot (CLI) ===")
    print("Nhập câu hỏi liên quan sức khỏe mắt / tư thế / mệt mỏi khi dùng máy tính.")
    print("Gõ 'exit' hoặc để trống rồi Enter để thoát.\n")

    thread_id = "cli_user"

    while True:
        try:
            user_input = input("Bạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nThoát chatbot.")
            break

        if not user_input or user_input.lower() in {"exit", "quit"}:
            print("Thoát chatbot.")
            break

        answer = chat_interface(user_input, thread_id=thread_id)
        print("Bot :", answer)
        print()


if __name__ == "__main__":
    main()


