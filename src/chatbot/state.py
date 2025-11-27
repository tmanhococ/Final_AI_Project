"""State definition for the LangGraph-based healthcare chatbot.

Theo DESIGN_DOCS_V2, state chia sẻ giữa các node được định nghĩa bằng TypedDict.
Ở đây ta tách riêng module state trong namespace chatbot.
"""

from __future__ import annotations

from typing import Annotated, List, Literal, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class GraphState(TypedDict):
    """Global graph state (V2.0) dùng cho toàn bộ LangGraph.

    Thuộc tính:
    - messages: lịch sử hội thoại (User + AI).
    - original_question: câu hỏi gốc mới nhất của user.
    - reformulated_question: câu hỏi đã được viết lại (Contextualize).
    - generation: câu trả lời cuối cùng.
    - analyzed_intent: loại intent RAG ('realtime_data', 'chunked_data', 'both', 'fall_back').
    - sub_queries: danh sách truy vấn con (cho CSV / retriever).
    - context: danh sách chuỗi context tích lũy từ các node.
    - retry_count: số lần đã retry qua rewriter.
    - answer_valid: cờ đánh dấu câu trả lời hiện tại có hợp lệ không (grader).
    """

    # --- CHAT MEMORY ---
    messages: Annotated[List[BaseMessage], add_messages]

    # --- INPUT/OUTPUT ---
    original_question: str
    reformulated_question: str
    generation: str

    # --- RAG INTERNAL CONTROL ---
    analyzed_intent: Literal["realtime_data", "chunked_data", "both", "fall_back"]
    sub_queries: List[str]
    context: List[str]
    csv_context: NotRequired[List[str]]
    doc_context: NotRequired[List[str]]
    route: NotRequired[Literal["social", "health"]]
    retry_count: int

    # --- GRADER FLAGS ---
    answer_valid: bool



