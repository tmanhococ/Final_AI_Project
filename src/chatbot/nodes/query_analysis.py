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
    """Heuristic đơn giản phân loại intent dựa trên từ khóa.
    
    Phân loại câu hỏi thành 4 loại:
    - "realtime_data": câu hỏi về dữ liệu thống kê/logs (dùng CSV agent)
    - "chunked_data": câu hỏi về kiến thức y tế (dùng vector retriever)
    - "both": câu hỏi cần cả CSV và documents
    - "fall_back": không xác định được (mặc định dùng chunked_data)
    
    Parameters
    ----------
    question:
        Câu hỏi đã được contextualize (chỉ câu hỏi mới nhất).
    
    Returns
    -------
    IntentLiteral
        Loại intent: "realtime_data" | "chunked_data" | "both" | "fall_back"
    """
    # Chuẩn hóa input: lowercase để so sánh không phân biệt hoa thường
    q = question.lower()
    
    # Danh sách từ khóa cho CSV queries (thống kê, logs, dữ liệu số)
    csv_keywords = [
        "log",
        "session",
        "thống kê",
        "summary",
        "csv",
        "phiên đo",
        "thời lượng",
        "dữ liệu",
        "bao nhiêu lần",
        "trung bình",
        "tổng",
        "số lượng",
        "phân tích",
        "duration",
        "avg",
        "average",
        "mean",
        "count",
    ]
    
    # Danh sách từ khóa cho document queries (kiến thức y tế, triệu chứng, điều trị)
    doc_keywords = [
        "bệnh",
        "triệu chứng",
        "dấu hiệu",
        "nguyên nhân",
        "điều trị",
        "hội chứng",
        "mỏi mắt",
        "đau",
        "nhức",
        "là gì",
        "giải thích",
        "phòng ngừa",
        "cách",
        "làm sao",
        "như thế nào",
        "tại sao",
        "cvs",
        "computer vision syndrome",
        "mắt",
        "thị lực",
        "bác sĩ",
        "tiến sĩ",
        "ts.bs",
        "giáo sư",
        "chuyên gia",
        "doctor",
        "professor",
        "expert",
        "researcher",
        "who is",
        "là ai",
        "ai là",
        "tiểu sử",
        "nói gì",
    ]

    # Kiểm tra xem câu hỏi có chứa từ khóa CSV hay doc
    has_csv = any(k in q for k in csv_keywords)
    has_doc = any(k in q for k in doc_keywords)

    # Routing logic: ưu tiên "both" nếu có cả hai, sau đó csv, sau đó doc
    if has_csv and has_doc:
        return "both"
    if has_csv:
        return "realtime_data"
    if has_doc:
        return "chunked_data"
    
    # Mặc định: ưu tiên khai thác docs thay vì fall_back
    # (vì hầu hết câu hỏi y tế đều có thể tìm trong documents)
    return "chunked_data"


def analyze_query_node(
    state: StateDict,
    llm: Optional[BaseLanguageModel] = None,
) -> Dict[str, object]:
    """Node phân tích câu hỏi: xác định intent và tạo sub_queries.
    
    Node này được gọi sau contextualize_node. Nó phân loại intent (CSV, documents, both)
    và tạo sub_queries để truyền cho các node tiếp theo (csv_node, retriever_node).
    
    QUAN TRỌNG: Node này RESET sub_queries (không append) để tránh tích lũy.
    
    Parameters
    ----------
    state:
        GraphState với field "reformulated_question" chứa câu hỏi đã được contextualize.
    llm:
        Optional LLM (hiện tại không dùng, có thể mở rộng để dùng LLM phân tích intent).
    
    Returns
    -------
    Dict[str, object]
        {
            "analyzed_intent": IntentLiteral - loại intent đã phân tích,
            "sub_queries": List[str] - danh sách câu hỏi con (RESET, không append)
        }
    
    Raises
    ------
    ValueError
        Nếu reformulated_question rỗng.
    """
    # Lấy câu hỏi đã được contextualize (CHỈ câu hỏi mới nhất)
    question = str(state.get("reformulated_question", "")).strip()
    
    # Validate input
    if not question:
        raise ValueError("analyze_query_node requires non-empty 'reformulated_question'.")
    
    # Phân loại intent bằng heuristic (có thể mở rộng dùng LLM sau)
    intent = heuristic_analyze_intent(question)
    
    # RESET sub_queries: tạo list mới với câu hỏi hiện tại
    # (có thể mở rộng để tách thành nhiều sub_queries nếu cần)
    sub_queries: List[str] = [question]
    
    # Trả về analyzed_intent và sub_queries mới (RESET)
    return {
        "analyzed_intent": intent,
        "sub_queries": sub_queries,
    }


