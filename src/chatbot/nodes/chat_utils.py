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

    Phân loại câu hỏi thành "social" (chào hỏi/xã giao) hoặc "health" (câu hỏi y tế).

    Parameters
    ----------
    text:
        Latest user message content.

    Returns
    -------
    Literal["social", "health"]
        "social" nếu là câu chào hỏi/xã giao, ngược lại "health".
    """
    # Chuẩn hóa input: lowercase và strip whitespace
    lowered = text.lower().strip()
    
    # Danh sách từ khóa xã giao/chào hỏi
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
    
    # Kiểm tra exact match cho các từ ngắn (như "hi", "hey", "hello")
    if lowered in ["hi", "hey", "hello"]:
        return "social"
    
    # Kiểm tra keyword matching: nếu có bất kỳ từ khóa xã giao nào trong text
    if any(k in lowered for k in social_keywords):
        return "social"
    
    # Mặc định: tất cả các câu hỏi khác đều là health-related
    return "health"


def guardrails_node(state: StateDict) -> Dict[str, object]:
    """Node guardrails: phân loại câu hỏi thành social hoặc health.
    
    Node này là điểm vào đầu tiên của graph, quyết định routing:
    - Nếu là câu chào hỏi/xã giao -> route="social" -> social_bot
    - Nếu là câu hỏi y tế -> route="health" -> contextualize -> query_analysis
    
    Parameters
    ----------
    state:
        GraphState với field "messages" chứa lịch sử hội thoại.
    
    Returns
    -------
    Dict[str, object]
        {"route": "social" | "health"}
    
    Raises
    ------
    ValueError
        Nếu messages rỗng hoặc message cuối không phải HumanMessage.
    """
    # Lấy danh sách messages từ state
    messages: List[BaseMessage] = state.get("messages", [])  # type: ignore[assignment]
    if not messages:
        raise ValueError("guardrails_node requires non-empty 'messages' in state.")

    # Lấy message cuối cùng (câu hỏi mới nhất của user)
    last = messages[-1]
    if not isinstance(last, HumanMessage):
        raise ValueError("Last message must be HumanMessage.")

    # Phân loại intent dựa trên nội dung message
    intent = detect_intent_from_text(str(last.content))
    
    # Trả về route để graph routing sử dụng
    return {"route": intent}


def social_response_node(state: StateDict, llm: BaseLanguageModel) -> Dict[str, object]:
    """Node trả lời các câu chào hỏi/xã giao (social bot).
    
    Node này chỉ được gọi khi guardrails phát hiện route="social".
    Sử dụng LLM để tạo câu trả lời thân thiện, ngắn gọn cho các câu chào hỏi.
    
    Parameters
    ----------
    state:
        GraphState với field "messages" chứa lịch sử hội thoại.
    llm:
        LLM instance để sinh câu trả lời.
    
    Returns
    -------
    Dict[str, object]
        {"generation": str} - câu trả lời đã được sinh.
    
    Raises
    ------
    ValueError
        Nếu messages rỗng.
    """
    # Lấy danh sách messages từ state
    messages: List[BaseMessage] = state.get("messages", [])  # type: ignore[assignment]
    if not messages:
        raise ValueError("social_response_node requires 'messages' in state.")

    # Tạo system prompt cho social bot
    system_prompt = (
        "Bạn là trợ lý sức khỏe thân thiện. "
        "Hãy trả lời ngắn gọn, lịch sự cho các câu chào hỏi/xã giao."
    )
    
    # Tạo conversation với system prompt + lịch sử hội thoại
    conversation: List[BaseMessage] = [
        SystemMessage(content=system_prompt),
        *messages,
    ]
    
    # Gọi LLM để sinh câu trả lời
    response = llm.invoke(conversation)
    
    # Trích xuất nội dung từ response (hỗ trợ cả BaseMessage và str)
    generation = (
        str(response.content) if hasattr(response, "content") else str(response)
    )

    # Trả về generation (chỉ field này, không trả về toàn bộ state để tránh duplicate messages)
    return {"generation": generation}





def contextualize_question(
    messages: Iterable[BaseMessage],
    original_question: str,
    llm: BaseLanguageModel,
) -> str:
    """Viết lại câu hỏi dựa trên lịch sử hội thoại (Contextualize).
    
    QUAN TRỌNG: Chỉ viết lại RIÊNG câu hỏi mới nhất, không tổng hợp hay liệt kê các câu hỏi trước.
    Mục đích là làm rõ câu hỏi hiện tại dựa trên context của cuộc hội thoại.

    Parameters
    ----------
    messages:
        Full chat history (user + assistant) để cung cấp context.
    original_question:
        Câu hỏi mới nhất của user cần được viết lại.
    llm:
        LLM dùng để rewrite. Trong test có thể dùng Dummy trả về chuỗi cố định.

    Returns
    -------
    str
        Câu hỏi đã được viết lại thành một câu độc lập, rõ nghĩa (CHỈ MỘT CÂU DUY NHẤT).
    """
    # Chuyển đổi messages thành text format để làm context
    history_texts: List[str] = []
    for m in messages:
        role = m.type
        history_texts.append(f"{role.upper()}: {m.content}")
    
    # Giới hạn lịch sử 10 messages gần nhất để tránh prompt quá dài
    history_block = "\n".join(history_texts[-10:])

    # Prompt được điều chỉnh rõ ràng: CHỈ viết lại câu hỏi mới nhất, không tổng hợp
    prompt = (
        "Dựa trên lịch sử hội thoại, hãy viết lại RIÊNG câu hỏi mới nhất "
        "thành một câu độc lập, rõ ràng. KHÔNG tổng hợp hay liệt kê các câu hỏi trước.\n\n"
        "Lịch sử:\n"
        f"{history_block}\n\n"
        f"Câu hỏi MỚI NHẤT cần viết lại: {original_question}\n\n"
        "Câu hỏi đã viết lại (CHỈ MỘT CÂU DUY NHẤT):"
    )
    
    # Gọi LLM để viết lại câu hỏi
    response = llm.invoke(prompt)
    
    # Trích xuất nội dung từ response
    text = str(response.content) if hasattr(response, "content") else str(response)
    
    # Trả về câu hỏi đã được viết lại (đã strip whitespace)
    return text.strip()


def contextualize_node(state: StateDict, llm: BaseLanguageModel) -> Dict[str, object]:
    """Node contextualize: viết lại câu hỏi dựa trên lịch sử hội thoại.
    
    Node này được gọi sau guardrails khi route="health".
    Mục đích: làm rõ câu hỏi hiện tại dựa trên context của cuộc hội thoại trước đó.
    
    Parameters
    ----------
    state:
        GraphState với fields:
        - "messages": lịch sử hội thoại
        - "original_question": câu hỏi gốc mới nhất
    llm:
        LLM instance để viết lại câu hỏi.
    
    Returns
    -------
    Dict[str, object]
        {"reformulated_question": str} - câu hỏi đã được viết lại.
    
    Raises
    ------
    ValueError
        Nếu messages rỗng hoặc original_question rỗng.
    """
    # Lấy messages và original_question từ state
    messages: List[BaseMessage] = state.get("messages", [])  # type: ignore[assignment]
    original_question = str(state.get("original_question", "")).strip()
    
    # Validate input
    if not messages or not original_question:
        raise ValueError(
            "contextualize_node requires 'messages' and non-empty 'original_question'."
        )
    
    # Gọi hàm core để viết lại câu hỏi
    rewritten = contextualize_question(messages, original_question, llm)
    
    # Trả về reformulated_question (chỉ field này, không trả về toàn bộ state)
    return {"reformulated_question": rewritten}


