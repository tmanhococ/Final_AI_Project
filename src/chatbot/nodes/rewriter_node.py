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
    """Dùng LLM (hoặc DummyLLM) để viết lại query tốt hơn."""
    prompt = (
        "Câu hỏi ban đầu có vẻ chưa được trả lời tốt hoặc chưa rõ ràng.\n"
        "Câu hỏi: "
        f"{question}\n"
        "Câu trả lời trước đó:\n"
        f"{last_answer}\n\n"
        "Hãy viết lại câu hỏi sao cho rõ ràng, cụ thể và dễ trả lời hơn."
    )
    response = llm.invoke(prompt)
    text = str(response.content) if hasattr(response, "content") else str(response)
    return text.strip()


def rewriter_node(state: StateDict, llm: BaseLanguageModel) -> Dict[str, object]:
    """Node viết lại query và tăng retry_count.

    Input:
    - state["reformulated_question"]: str
    - state["generation"]: str (câu trả lời trước)
    - state["retry_count"]: int

    Output:
    - state["sub_queries"]: List[str] (mới)
    - state["retry_count"]: int (tăng +1)
    """
    question = str(state.get("reformulated_question", "")).strip()
    last_answer = str(state.get("generation", "")).strip()
    retry_count = int(state.get("retry_count", 0))

    if not question:
        raise ValueError("rewriter_node requires non-empty 'reformulated_question'.")

    new_query = rewrite_query(question, last_answer, llm)

    return {
        "sub_queries": [new_query],
        "retry_count": retry_count + 1,
    }


