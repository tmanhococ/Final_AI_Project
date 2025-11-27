"""
Tests for high-level `chat_interface` function.

Sử dụng LLM thật (Gemini) thông qua `chat_interface` mặc định
để kiểm tra end-to-end pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chatbot.chat_interface import chat_interface  # noqa: E402
from src.chatbot.config import CHATBOT_CONFIG  # noqa: E402


def test_chat_interface_with_real_llm() -> None:
    """Gọi trực tiếp chat_interface sử dụng LLM thật (Gemini)."""
    # Đảm bảo cấu hình hợp lệ (có GOOGLE_API_KEY)
    CHATBOT_CONFIG.validate()

    question = "Tôi bị mỏi mắt khi làm việc với máy tính, tôi nên làm gì?"
    answer = chat_interface(question, thread_id="test_user_real")

    print("chat_interface question:", question.encode("utf-8", errors="ignore"))
    print("chat_interface answer:", str(answer).encode("utf-8", errors="ignore"))

    assert isinstance(answer, str) and len(answer) > 0



