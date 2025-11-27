"""High-level chat interface for the healthcare chatbot.

Hàm `chat_interface` được thiết kế gần giống ví dụ trong DESIGN_DOCS_V2:

    from src.chatbot.chat_interface import chat_interface
    answer = chat_interface("Tôi bị mỏi mắt khi làm việc với máy tính", thread_id="user1")

Mặc định sẽ khởi tạo app thật (Gemini + Chroma + CSV agent). Trong unit test,
có thể truyền vào app giả (graph dùng DummyLLM) qua tham số `app`.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.messages import HumanMessage

from src.chatbot.app_runtime import create_chatbot_app
from src.chatbot.state import GraphState


_GLOBAL_APP: Any | None = None


def _get_or_create_app() -> Any:
    """Lazy-init global app cho production."""
    global _GLOBAL_APP
    if _GLOBAL_APP is None:
        _GLOBAL_APP = create_chatbot_app()
    return _GLOBAL_APP


def chat_interface(
    user_input: str,
    thread_id: str = "local_user_1",
    *,
    app: Optional[Any] = None,
) -> str:
    """Send a message to the chatbot and get contextual response.

    Parameters
    ----------
    user_input:
        Câu hỏi / tin nhắn của người dùng.
    thread_id:
        ID phiên chat (dùng cho MemorySaver của LangGraph).
    app:
        Optional compiled LangGraph app; nếu None sẽ gọi `create_chatbot_app()`.

    Returns
    -------
    str
        Câu trả lời cuối cùng của chatbot.
    """
    graph_app = app or _get_or_create_app()

    # Config cho MemorySaver
    config = {"configurable": {"thread_id": thread_id}}

    # Khởi tạo state tối thiểu
    inputs: GraphState = {
        "messages": [HumanMessage(content=user_input)],
        "original_question": user_input,
        "reformulated_question": "",
        "generation": "",
        "analyzed_intent": "fall_back",
        "sub_queries": [],
        "context": [],
        "retry_count": 0,
        "answer_valid": True,
    }

    try:
        result: GraphState = graph_app.invoke(inputs, config=config)
        return result.get("generation", "Xin lỗi, tôi không thể trả lời lúc này.")
    except Exception as exc:  # pragma: no cover - path lỗi thực tế
        print("chat_interface error:", repr(exc))
        return "Hệ thống đang gặp sự cố kỹ thuật."



