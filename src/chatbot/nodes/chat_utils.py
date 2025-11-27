"""Chat-related utility nodes: Guardrails, Social Bot, Contextualize.

Đây là lớp "vỏ hội thoại" bao quanh lõi RAG, theo DESIGN_DOCS_V2.

Thiết kế:
- Các hàm core là pure function, thao tác trên dict state để dễ test.
- LLM được truyền vào từ ngoài (dependency injection) để test bằng DummyLLM.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Literal, Mapping, MutableMapping

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage


StateDict = MutableMapping[str, object]


def detect_intent_from_text(text: str) -> Literal["social", "health"]:
    """Simple heuristic-based intent detection for guardrails.

    Parameters
    ----------
    text:
        Latest user message content.

    Returns
    -------
    Literal["social", "health"]
        "social" nếu là câu chào hỏi/xã giao, ngược lại "health".
    """
    lowered = text.lower().strip()
    # Mở rộng keywords để bắt được nhiều trường hợp chào hỏi hơn
    social_keywords = [
        "hi",
        "hello",
        "hey",
        "chào",
        "xin chào",
        "cảm ơn",
        "thank",
        "thanks",
        "bạn là ai",
        "you there",
        "tạm biệt",
        "bye",
        "goodbye",
        "khỏe không",
        "bạn khỏe",
    ]
    # Kiểm tra exact match cho các từ ngắn (như "hi", "hey")
    if lowered in ["hi", "hey", "hello"]:
        return "social"
    # Kiểm tra keyword matching
    if any(k in lowered for k in social_keywords):
        return "social"
    return "health"


def guardrails_node(state: StateDict) -> Dict[str, object]:
    messages: List[BaseMessage] = state.get("messages", [])  # type: ignore[assignment]
    if not messages:
        raise ValueError("guardrails_node requires non-empty 'messages' in state.")

    last = messages[-1]
    if not isinstance(last, HumanMessage):
        raise ValueError("Last message must be HumanMessage.")

    intent = detect_intent_from_text(str(last.content))
    return {"route": intent}


def social_response_node(state: StateDict, llm: BaseLanguageModel) -> Dict[str, object]:
    messages: List[BaseMessage] = state.get("messages", [])  # type: ignore[assignment]
    if not messages:
        raise ValueError("social_response_node requires 'messages' in state.")

    system_prompt = (
        "Bạn là trợ lý sức khỏe thân thiện. "
        "Hãy trả lời ngắn gọn, lịch sự cho các câu chào hỏi/xã giao."
    )
    conversation: List[BaseMessage] = [
        SystemMessage(content=system_prompt),
        *messages,
    ]
    response = llm.invoke(conversation)
    generation = (
        str(response.content) if hasattr(response, "content") else str(response)
    )

    return {"generation": generation}





def contextualize_question(
    messages: Iterable[BaseMessage],
    original_question: str,
    llm: BaseLanguageModel,
) -> str:
    """Viết lại câu hỏi dựa trên lịch sử hội thoại (Contextualize).

    Parameters
    ----------
    messages:
        Full chat history (user + assistant).
    original_question:
        Câu hỏi mới nhất của user.
    llm:
        LLM dùng để rewrite. Trong test có thể dùng Dummy trả về chuỗi cố định.

    Returns
    -------
    str
        Câu hỏi đã được viết lại thành một câu độc lập, rõ nghĩa.
    """
    history_texts: List[str] = []
    for m in messages:
        role = m.type
        history_texts.append(f"{role.upper()}: {m.content}")
    history_block = "\n".join(history_texts[-10:])  # giới hạn cho gọn

    prompt = (
        "Dựa trên lịch sử hội thoại sau và câu hỏi cuối cùng, "
        "hãy viết lại câu hỏi thành một câu độc lập, đầy đủ chủ ngữ vị ngữ.\n\n"
        "Lịch sử:\n"
        f"{history_block}\n\n"
        f"Câu hỏi mới: {original_question}\n\n"
        "Câu hỏi đã viết lại:"
    )
    response = llm.invoke(prompt)
    text = str(response.content) if hasattr(response, "content") else str(response)
    return text.strip()


def contextualize_node(state: StateDict, llm: BaseLanguageModel) -> Dict[str, object]:
    messages: List[BaseMessage] = state.get("messages", [])  # type: ignore[assignment]
    original_question = str(state.get("original_question", "")).strip()
    if not messages or not original_question:
        raise ValueError(
            "contextualize_node requires 'messages' and non-empty 'original_question'."
        )
    
    rewritten = contextualize_question(messages, original_question, llm)
    return {"reformulated_question": rewritten}


