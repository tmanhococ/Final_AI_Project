"""Rewrite node: viết lại truy vấn khi kết quả không tốt / cần retry."""

from __future__ import annotations

from typing import Dict, List, MutableMapping

from langchain_core.language_models import BaseLanguageModel


StateDict = MutableMapping[str, object]


def rewrite_query(
    question: str,
    last_answer: str,
    llm: BaseLanguageModel,
) -> str:
    """Dùng LLM để viết lại query tốt hơn khi câu trả lời trước không hợp lệ.
    
    Hàm này được gọi trong retry loop khi answer_grader phát hiện câu trả lời không hợp lệ.
    Mục đích: cải thiện câu hỏi để có cơ hội nhận được câu trả lời tốt hơn.
    
    Parameters
    ----------
    question:
        Câu hỏi hiện tại (reformulated_question) cần được viết lại.
    last_answer:
        Câu trả lời trước đó (generation) không hợp lệ, dùng làm context.
    llm:
        LLM instance để viết lại câu hỏi.
    
    Returns
    -------
    str
        Câu hỏi đã được viết lại, rõ ràng và cụ thể hơn.
    """
    # Tạo prompt để LLM viết lại câu hỏi
    prompt = (
        "Câu hỏi ban đầu có vẻ chưa được trả lời tốt hoặc chưa rõ ràng.\n"
        "Câu hỏi: "
        f"{question}\n"
        "Câu trả lời trước đó:\n"
        f"{last_answer}\n\n"
        "Hãy viết lại câu hỏi sao cho rõ ràng, cụ thể và dễ trả lời hơn."
    )
    
    # Gọi LLM để viết lại câu hỏi
    response = llm.invoke(prompt)
    
    # Trích xuất nội dung từ response
    text = str(response.content) if hasattr(response, "content") else str(response)
    
    # Trả về câu hỏi đã được viết lại (đã strip whitespace)
    return text.strip()


def rewriter_node(state: StateDict, llm: BaseLanguageModel) -> Dict[str, object]:
    """Node viết lại query và tăng retry_count khi câu trả lời không hợp lệ.
    
    Node này được gọi trong retry loop khi answer_grader phát hiện answer_valid=False.
    Mục đích: viết lại câu hỏi để có cơ hội nhận được câu trả lời tốt hơn ở lần thử tiếp theo.
    
    QUAN TRỌNG: Node này RESET sub_queries (không append) và tăng retry_count.
    
    Parameters
    ----------
    state:
        GraphState với fields:
        - "reformulated_question": str - câu hỏi hiện tại cần viết lại
        - "generation": str - câu trả lời trước đó không hợp lệ
        - "retry_count": int - số lần đã retry
    llm:
        LLM instance để viết lại câu hỏi.
    
    Returns
    -------
    Dict[str, object]
        {
            "sub_queries": List[str] - câu hỏi mới đã được viết lại (RESET),
            "retry_count": int - retry_count + 1
        }
    
    Raises
    ------
    ValueError
        Nếu reformulated_question rỗng.
    """
    # Lấy các fields từ state
    question = str(state.get("reformulated_question", "")).strip()
    last_answer = str(state.get("generation", "")).strip()
    retry_count = int(state.get("retry_count", 0))

    # Validate input
    if not question:
        raise ValueError("rewriter_node requires non-empty 'reformulated_question'.")

    # Gọi hàm core để viết lại câu hỏi
    new_query = rewrite_query(question, last_answer, llm)

    # RESET sub_queries với câu hỏi mới và tăng retry_count
    return {
        "sub_queries": [new_query],
        "retry_count": retry_count + 1,
    }


