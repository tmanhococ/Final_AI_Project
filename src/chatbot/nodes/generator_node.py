"""Generator node: tổng hợp context để sinh câu trả lời cuối cùng."""

from __future__ import annotations

from typing import Dict, List, MutableMapping

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage


StateDict = MutableMapping[str, object]


def generator_node(state: StateDict, llm: BaseLanguageModel) -> Dict[str, object]:
    """Sinh câu trả lời dựa trên context + câu hỏi.
    
    Node này là node cuối cùng trong RAG pipeline, tổng hợp context từ CSV và documents
    để tạo câu trả lời cho người dùng. QUAN TRỌNG: Chỉ sử dụng câu hỏi mới nhất (reformulated_question).

    Parameters
    ----------
    state:
        GraphState với fields:
        - "reformulated_question": str - câu hỏi đã được contextualize (CHỈ câu hỏi mới nhất)
        - "context": List[str] - context đã được merge từ csv_context + filtered_docs (ưu tiên)
        - "csv_context": List[str] - kết quả phân tích CSV (fallback nếu context rỗng)
        - "doc_context": List[str] - documents đã retrieve (fallback nếu context rỗng)
    llm:
        LLM instance để sinh câu trả lời.

    Returns
    -------
    Dict[str, object]
        {"generation": str} - câu trả lời đã được sinh.
    
    Raises
    ------
    ValueError
        Nếu reformulated_question rỗng.
    """
    # Lấy câu hỏi đã được contextualize (CHỈ câu hỏi mới nhất, không phải tổng hợp)
    question = str(state.get("reformulated_question", "")).strip()
    
    # Lấy context từ state (ưu tiên context đã được merge từ doc_grader)
    context_list: List[str] = list(state.get("context", []) or [])  # type: ignore[arg-type]
    
    # Fallback: nếu context rỗng, dùng csv_context + doc_context
    csv_context: List[str] = list(state.get("csv_context", []) or [])  # type: ignore[arg-type]
    doc_context: List[str] = list(state.get("doc_context", []) or [])  # type: ignore[arg-type]

    # Validate input
    if not question:
        raise ValueError("generator_node requires non-empty 'reformulated_question'.")

    # Chọn effective_context: ưu tiên context_list (đã được merge và lọc), fallback về csv+doc
    effective_context = context_list or (csv_context + doc_context)

    # Nếu không có context, trả về message mặc định
    if not effective_context:
        generation = (
            "Xin lỗi, tôi không có đủ thông tin đáng tin cậy để trả lời câu hỏi này. "
            "Bạn có thể cung cấp thêm dữ liệu cụ thể hơn hoặc hỏi một câu khác?"
        )
    else:
        # Nối các context blocks thành một chuỗi
        context_block = "\n\n".join(effective_context)
        
        # Tạo system prompt cho LLM
        system_prompt = (
            "Bạn là trợ lý sức khỏe. Sử dụng thông tin trong phần CONTEXT để trả lời "
            "câu hỏi của người dùng một cách rõ ràng, an toàn, không chẩn đoán quá mức.\n"
            "Nếu thông tin không đủ, hãy nói rõ là bạn không chắc chắn."
        )
        
        # Tạo user content với context và question
        user_content = f"CONTEXT:\n{context_block}\n\nQUESTION:\n{question}"

        # Gọi LLM để sinh câu trả lời
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
        response = llm.invoke(messages)
        
        # Trích xuất nội dung từ response
        generation = (
            str(response.content) if hasattr(response, "content") else str(response)
        )

    # Trả về generation (chỉ field này, không trả về toàn bộ state)
    return {"generation": generation}


