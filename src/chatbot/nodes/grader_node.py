"""Grader nodes: chấm độ liên quan của document và kiểm tra câu trả lời."""

from __future__ import annotations

from typing import Dict, List, MutableMapping


StateDict = MutableMapping[str, object]


def document_relevance_grader(doc_text: str, question: str) -> bool:
    """Heuristic đơn giản kiểm tra doc có liên quan câu hỏi không.
    
    Hàm này dùng để lọc các documents không liên quan trước khi đưa vào generator.
    Logic: True nếu có ít nhất 1 từ khóa (từ dài hơn 4 ký tự) trùng trong doc.
    
    Parameters
    ----------
    doc_text:
        Nội dung của document cần kiểm tra.
    question:
        Câu hỏi đã được contextualize (chỉ câu hỏi mới nhất).
    
    Returns
    -------
    bool
        True nếu document liên quan, False nếu không liên quan.
    """
    # Trích xuất các từ khóa từ câu hỏi (chỉ lấy từ dài hơn 4 ký tự để tránh noise)
    q_tokens = {t for t in question.lower().split() if len(t) > 4}
    
    # Trích xuất các từ từ document
    d_tokens = set(doc_text.lower().split())
    
    # Kiểm tra overlap: nếu có ít nhất 1 từ khóa trùng -> liên quan
    return bool(q_tokens & d_tokens)


def answer_quality_grader(answer: str, question: str, context: str) -> bool:
    """Heuristic đơn giản kiểm tra câu trả lời có "hợp lý" không.
    
    Hàm này dùng để phát hiện các câu trả lời kém chất lượng, trigger retry loop.
    Logic: False nếu answer quá ngắn hoặc chỉ lặp lại câu hỏi.
    
    Parameters
    ----------
    answer:
        Câu trả lời đã được sinh (generation).
    question:
        Câu hỏi đã được contextualize (chỉ câu hỏi mới nhất).
    context:
        Context đã được sử dụng để sinh câu trả lời (không dùng trong logic hiện tại).
    
    Returns
    -------
    bool
        True nếu câu trả lời hợp lệ, False nếu cần retry.
    """
    # Chuẩn hóa answer: lowercase và strip
    a = answer.strip().lower()
    
    # Kiểm tra: nếu answer quá ngắn (< 10 ký tự) -> không hợp lệ
    if len(a) < 10:
        return False
    
    # Kiểm tra: nếu answer chỉ lặp lại câu hỏi -> không hợp lệ
    if a == question.strip().lower():
        return False
    
    # Mặc định: hợp lệ
    return True


def doc_grader_node(state: StateDict) -> Dict[str, object]:
    """Node chấm độ liên quan của docs trong context.
    
    Node này lọc các documents không liên quan và merge csv_context + doc_context
    vào context chính. QUAN TRỌNG: Node này RESET context (không append) để tránh tích lũy.

    Parameters
    ----------
    state:
        GraphState với fields:
        - "doc_context": List[str] - documents đã retrieve
        - "csv_context": List[str] - kết quả phân tích CSV (được giữ nguyên, không lọc)
        - "reformulated_question": str - câu hỏi đã được contextualize

    Returns
    -------
    Dict[str, object]
        {
            "doc_context": List[str] - docs đã lọc (chỉ giữ lại docs liên quan),
            "context": List[str] - csv_context + filtered_docs (RESET, không append)
        }
    
    Raises
    ------
    ValueError
        Nếu reformulated_question rỗng.
    """
    # Lấy doc_context và csv_context từ state
    doc_context: List[str] = list(state.get("doc_context", []))  # type: ignore[assignment]
    csv_context: List[str] = list(state.get("csv_context", []))  # type: ignore[assignment]
    question = str(state.get("reformulated_question", "")).strip()
    
    # Validate input
    if not question:
        raise ValueError("doc_grader_node requires non-empty 'reformulated_question'.")

    # Lọc documents: chỉ giữ lại những documents liên quan đến câu hỏi
    filtered: List[str] = []
    for block in doc_context:
        if document_relevance_grader(block, question):
            filtered.append(block)

    # RESET context: merge csv_context (không lọc) + filtered_docs
    # Không append vào context cũ để tránh tích lũy
    final_context = csv_context + filtered

    # Trả về doc_context đã lọc và context mới (RESET)
    return {
        "doc_context": filtered,
        "context": final_context,
    }


def answer_grader_node(state: StateDict) -> Dict[str, object]:
    """Node chấm câu trả lời để phát hiện câu trả lời quá kém/chưa hợp lý.
    
    Node này được gọi sau generator_node để kiểm tra chất lượng câu trả lời.
    Nếu answer_valid=False và retry_count < max_retries, graph sẽ quay lại rewriter_node.
    
    Parameters
    ----------
    state:
        GraphState với fields:
        - "generation": str - câu trả lời đã được sinh
        - "reformulated_question": str - câu hỏi đã được contextualize
        - "context": List[str] - context đã được sử dụng
    
    Returns
    -------
    Dict[str, object]
        {"answer_valid": bool} - True nếu câu trả lời hợp lệ, False nếu cần retry.
    """
    # Lấy các fields từ state
    answer = str(state.get("generation", "")).strip()
    question = str(state.get("reformulated_question", "")).strip()
    context_list: List[str] = list(state.get("context", []))  # type: ignore[assignment]
    
    # Nối context thành chuỗi (để truyền vào grader, mặc dù hiện tại không dùng)
    context_joined = "\n".join(context_list)

    # Gọi hàm core để kiểm tra chất lượng câu trả lời
    is_valid = answer_quality_grader(answer, question, context_joined)

    # Trả về answer_valid (chỉ field này, không trả về toàn bộ state)
    return {"answer_valid": is_valid}


