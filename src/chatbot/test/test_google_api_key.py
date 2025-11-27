r"""
Simple manual test to verify GOOGLE_API_KEY works with Google Gemini.

Cách dùng:
    1) Đảm bảo đã kích hoạt venv chatbot:
           cd D:\AI_Final\Final_AI_Project
           .\.venv_chatbot\Scripts\Activate.ps1

    2) Đảm bảo đã cấu hình GOOGLE_API_KEY trong src/chatbot/.env hoặc .env root.

    3) Chạy:
           python -m src.chatbot.test_google_api_key

    Script sẽ:
    - In model đang dùng.
    - Gửi 1 prompt rất ngắn tới Gemini.
    - In ra câu trả lời hoặc thông báo lỗi chi tiết.
"""

from __future__ import annotations

import sys
from pathlib import Path

import google.generativeai as genai

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chatbot.config import CHATBOT_CONFIG  # noqa: E402


def main() -> None:
    print("=== GOOGLE_API_KEY connectivity test ===")
    print("LLM model name from config:", CHATBOT_CONFIG.llm_model_name)

    if not CHATBOT_CONFIG.google_api_key:
        print(
            "ERROR: GOOGLE_API_KEY is not set in environment.\n"
            "Hãy kiểm tra file .env (src/chatbot/.env hoặc .env ở project root)."
        )
        sys.exit(1)

    # Cấu hình client
    genai.configure(api_key=CHATBOT_CONFIG.google_api_key)

    try:
        model = genai.GenerativeModel(CHATBOT_CONFIG.llm_model_name)
        prompt = "Xin chào, hãy trả lời 1 câu rất ngắn: 'OK' nếu bạn nhận được yêu cầu."
        print("Sending test request to Gemini...")
        response = model.generate_content(prompt)
        text = getattr(response, "text", None) or getattr(response, "candidates", None)

        print("=== RAW RESPONSE OBJECT TYPE ===")
        print(type(response))
        print("=== RAW RESPONSE (truncated) ===")
        print(str(response)[:500])

        print("=== EXTRACTED TEXT (if available) ===")
        # Nhiều phiên bản google-generativeai cung cấp .text
        if hasattr(response, "text"):
            print(response.text)
        else:
            print("Không tìm thấy thuộc tính .text trên response, hãy xem RAW RESPONSE ở trên.")

        print("\nGoogle API key test: SUCCESS.")
    except Exception as exc:  # pragma: no cover - test thủ công
        print("\nGoogle API key test: FAILED.")
        print("Error type:", type(exc))
        print("Error detail:", repr(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()


