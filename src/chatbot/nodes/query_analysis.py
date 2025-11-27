"""Query analysis node: phân loại intent và tạo sub-queries.

Trong bản production thiết kế dùng LLM trả về JSON; ở đây để dễ test và
tránh gọi API, ta hiện thực heuristic đơn giản + chỗ hook cho LLM nếu cần.
"""

from __future__ import annotations

from typing import Dict, List, Literal, MutableMapping, Optional

from langchain_core.language_models import BaseLanguageModel


StateDict = MutableMapping[str, object]
IntentLiteral = Literal["realtime_data", "chunked_data", "both", "fall_back"]


def heuristic_analyze_intent(question: str) -> IntentLiteral:
    """Heuristic đơn giản phân loại intent dựa trên từ khóa."""
    q = question.lower()
    # Mở rộng keywords để bắt được ngôn ngữ tự nhiên hơn
    csv_keywords = [
        "log", "session", "thống kê", "summary", "csv",
        "phiên đo", "thời lượng", "dữ liệu", "bao nhiêu lần",
        "trung bình", "tổng", "số lượng", "thống kê", "phân tích",
        "duration", "avg", "average", "mean", "count"
    ]
    doc_keywords = [
        "bệnh", "triệu chứng", "dấu hiệu", "nguyên nhân", "điều trị",
        "hội chứng", "mỏi mắt", "đau", "nhức", "là gì", "giải thích",
        "phòng ngừa", "cách", "làm sao", "như thế nào", "tại sao",
        "cvs", "computer vision syndrome", "mắt", "thị lực"
    ]

    has_csv = any(k in q for k in csv_keywords)
    has_doc = any(k in q for k in doc_keywords)

    if has_csv and has_doc:
        return "both"
    if has_csv:
        return "realtime_data"
    if has_doc:
        return "chunked_data"
    return "fall_back"


def analyze_query_node(
    state: StateDict,
    llm: Optional[BaseLanguageModel] = None,
) -> Dict[str, object]:
    question = str(state.get("reformulated_question", "")).strip()
    if not question:
        raise ValueError("analyze_query_node requires non-empty 'reformulated_question'.")
    
    intent = heuristic_analyze_intent(question)
    sub_queries: List[str] = [question]
    
    
    return {
        "analyzed_intent": intent,
        "sub_queries": sub_queries,
    }


