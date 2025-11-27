"""Grader nodes: chấm độ liên quan của document và kiểm tra câu trả lời."""

from __future__ import annotations

from typing import Dict, List, MutableMapping


StateDict = MutableMapping[str, object]


def document_relevance_grader(doc_text: str, question: str) -> bool:
    """Heuristic đơn giản kiểm tra doc có liên quan câu hỏi không.

    Ở đây chỉ dùng:
    - True nếu có ít nhất 1 từ khóa (từ dài hơn 4 ký tự) trùng trong doc.
    """
    q_tokens = {t for t in question.lower().split() if len(t) > 4}
    d_tokens = set(doc_text.lower().split())
    return bool(q_tokens & d_tokens)


def answer_quality_grader(answer: str, question: str, context: str) -> bool:
    """Heuristic đơn giản kiểm tra câu trả lời có "hợp lý" không.

    - Trả về False nếu answer quá ngắn (< 10 ký tự) hoặc chỉ lặp lại câu hỏi.
    """
    a = answer.strip().lower()
    if len(a) < 10:
        return False
    if a == question.strip().lower():
        return False
    return True


def doc_grader_node(state: StateDict) -> Dict[str, object]:
    """Node chấm độ liên quan của docs trong context.

    Input:
    - state["doc_context"]: List[str] (block text docs)
    - state["csv_context"]: List[str] (được giữ nguyên)
    - state["reformulated_question"]: str

    Output:
    - state["context"]: csv_context + docs đã lọc
    - state["doc_context"]: docs đã lọc (để dùng cho vòng sau nếu cần)
    """
    doc_context: List[str] = list(state.get("doc_context", []))  # type: ignore[assignment]
    csv_context: List[str] = list(state.get("csv_context", []))  # type: ignore[assignment]
    question = str(state.get("reformulated_question", "")).strip()
    if not question:
        raise ValueError("doc_grader_node requires non-empty 'reformulated_question'.")

    filtered: List[str] = []
    for block in doc_context:
        if document_relevance_grader(block, question):
            filtered.append(block)

    final_context = csv_context + filtered

    new_state: Dict[str, object] = dict(state)
    new_state["doc_context"] = filtered
    new_state["context"] = final_context
    return new_state


def answer_grader_node(state: StateDict) -> Dict[str, object]:
    """Node chấm câu trả lời để phát hiện câu trả lời quá kém/chưa hợp lý.

    Input:
    - state["generation"]: str
    - state["reformulated_question"]: str
    - state["context"]: List[str]

    Output:
    - "answer_valid": bool
    """
    answer = str(state.get("generation", "")).strip()
    question = str(state.get("reformulated_question", "")).strip()
    context_list: List[str] = list(state.get("context", []))  # type: ignore[assignment]
    context_joined = "\n".join(context_list)

    is_valid = answer_quality_grader(answer, question, context_joined)

    new_state: Dict[str, object] = dict(state)
    new_state["answer_valid"] = is_valid
    return new_state


