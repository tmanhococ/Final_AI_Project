"""Generator node: tổng hợp context để sinh câu trả lời cuối cùng."""

from __future__ import annotations

from typing import Dict, List, MutableMapping

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage


StateDict = MutableMapping[str, object]


def generator_node(state: StateDict, llm: BaseLanguageModel) -> Dict[str, object]:
    """Sinh câu trả lời dựa trên context + câu hỏi.

    Input:
    - state["reformulated_question"]: str
    - state["context"]: List[str]

    Output:
    - state["generation"]: str
    """
    question = str(state.get("reformulated_question", "")).strip()
    context_list: List[str] = list(state.get("context", []) or [])  # type: ignore[arg-type]
    csv_context: List[str] = list(state.get("csv_context", []) or [])  # type: ignore[arg-type]
    doc_context: List[str] = list(state.get("doc_context", []) or [])  # type: ignore[arg-type]

    if not question:
        raise ValueError("generator_node requires non-empty 'reformulated_question'.")

    effective_context = context_list or (csv_context + doc_context)

    if not effective_context:
        generation = (
            "Xin lỗi, tôi không có đủ thông tin đáng tin cậy để trả lời câu hỏi này. "
            "Bạn có thể cung cấp thêm dữ liệu cụ thể hơn hoặc hỏi một câu khác?"
        )
    else:
        context_block = "\n\n".join(effective_context)
        system_prompt = (
            "Bạn là trợ lý sức khỏe. Sử dụng thông tin trong phần CONTEXT để trả lời "
            "câu hỏi của người dùng một cách rõ ràng, an toàn, không chẩn đoán quá mức.\n"
            "Nếu thông tin không đủ, hãy nói rõ là bạn không chắc chắn."
        )
        user_content = f"CONTEXT:\n{context_block}\n\nQUESTION:\n{question}"

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
        response = llm.invoke(messages)
        generation = (
            str(response.content) if hasattr(response, "content") else str(response)
        )

    return {"generation": generation}


