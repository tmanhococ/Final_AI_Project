"""LangGraph assembly for the healthcare Adaptive RAG chatbot.

Graph này hiện thực gần theo DESIGN_DOCS_V2:

Start -> Guardrails
  - route=social  -> SocialBot -> END
  - route=health  -> Contextualize -> QueryAnalysis
        -> (theo analyzed_intent) CSV / Retriever / Both
        -> DocGrader -> Generator -> AnswerGrader
        -> nếu answer_valid=False và retry_count < MAX_RETRIES -> Rewriter -> QueryAnalysis (loop)
           ngược lại -> END

Để thuận tiện cho test và linh hoạt, ta xây dựng một hàm build_graph nhận
các dependency (LLM, vector_store, csv_agent) từ bên ngoài.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.chatbot.config import CHATBOT_CONFIG
from src.chatbot.nodes.chat_utils import (
    contextualize_node,
    guardrails_node,
    social_response_node,
)
from src.chatbot.nodes.csv_node import csv_analyst_node
from src.chatbot.nodes.generator_node import generator_node
from src.chatbot.nodes.grader_node import answer_grader_node, doc_grader_node
from src.chatbot.nodes.query_analysis import analyze_query_node
from src.chatbot.nodes.retriever_node import medical_retriever_node
from src.chatbot.nodes.rewriter_node import rewriter_node
from src.chatbot.state import GraphState


def build_graph(
    *,
    llm,
    vector_store,
    csv_agent,
    max_retries: int | None = None,
):
    """Build LangGraph app for the chatbot.

    Parameters
    ----------
    llm:
        LLM dùng cho social/contextualize/generator/rewriter.
    vector_store:
        Chroma (hoặc vector store tương thích) cho medical docs.
    csv_agent:
        Pandas DataFrame agent cho summary/logs (đối tượng có .invoke()).
    max_retries:
        Số lần tối đa cho vòng lặp rewrite; nếu None dùng từ CHATBOT_CONFIG.

    Returns
    -------
    Compiled app (langgraph graph callable)
    """
    # Lấy max_retries từ parameter hoặc config
    retries_limit = max_retries if max_retries is not None else CHATBOT_CONFIG.max_retries

    # Tạo StateGraph với GraphState làm state schema
    workflow = StateGraph(GraphState)

    # --- Node wrappers với dependency injection ---
    # Thêm tất cả các nodes vào graph, dùng partial để inject dependencies (llm, vector_store, csv_agent)
    workflow.add_node("guardrails", guardrails_node)  # Phân loại social vs health
    workflow.add_node("social_bot", partial(social_response_node, llm=llm))  # Trả lời câu chào hỏi
    workflow.add_node("contextualize", partial(contextualize_node, llm=llm))  # Viết lại câu hỏi dựa trên context
    workflow.add_node("query_analysis", analyze_query_node)  # Phân loại intent (CSV, docs, both)
    workflow.add_node("csv_node", partial(csv_analyst_node, agent=csv_agent))  # Truy vấn CSV
    workflow.add_node("retriever_node", partial(medical_retriever_node, vector_store=vector_store))  # Retrieve documents
    workflow.add_node("doc_grader", doc_grader_node)  # Lọc documents không liên quan
    workflow.add_node("generator", partial(generator_node, llm=llm))  # Sinh câu trả lời
    workflow.add_node("answer_grader", answer_grader_node)  # Kiểm tra chất lượng câu trả lời
    workflow.add_node("rewriter", partial(rewriter_node, llm=llm))  # Viết lại query khi retry

    # --- Edges ---
    # Đặt guardrails làm entry point (node đầu tiên được gọi)
    workflow.set_entry_point("guardrails")

    # Guardrails routing: route social vs health
    def route_from_guardrails(state: GraphState) -> str:
        """Routing function: quyết định đi theo social path hay health path."""
        route = state.get("route", "health")  # type: ignore[assignment]
        if route == "social":
            return "social_bot"  # Đi đến social bot
        return "contextualize"  # Đi đến health path

    # Thêm conditional edge từ guardrails: routing dựa trên route (social/health)
    workflow.add_conditional_edges(
        "guardrails",
        route_from_guardrails,
        {
            "social_bot": "social_bot",
            "contextualize": "contextualize",
        },
    )

    # Social path: sau social_bot -> kết thúc (END)
    workflow.add_edge("social_bot", END)

    # Health path: contextualize -> query_analysis
    workflow.add_edge("contextualize", "query_analysis")

    # Query analysis routing: quyết định dùng CSV, retriever, cả hai, hay không dùng RAG
    def route_from_query_analysis(state: GraphState) -> str:
        """Routing function: quyết định đi theo CSV path, retriever path, both, hay no_rag."""
        intent = state.get("analyzed_intent", "fall_back")  # type: ignore[assignment]
        if intent == "realtime_data":
            return "csv_only"  # Chỉ dùng CSV
        if intent == "chunked_data":
            return "retriever_only"  # Chỉ dùng retriever
        if intent == "both":
            return "both"  # Dùng cả CSV và retriever
        # fall_back: chỉ dùng generator với context hiện tại (có thể rỗng)
        return "no_rag"

    # Thêm conditional edge từ query_analysis: routing dựa trên analyzed_intent
    workflow.add_conditional_edges(
        "query_analysis",
        route_from_query_analysis,
        {
            "csv_only": "csv_node",  # Chỉ dùng CSV -> csv_node
            "retriever_only": "retriever_node",  # Chỉ dùng retriever -> retriever_node
            "both": "csv_node",  # Cả hai -> csv_node trước, sau đó sẽ nối sang retriever_node
            "no_rag": "generator",  # Không dùng RAG -> generator trực tiếp
        },
    )

    # CSV node routing: sau csv_node, quyết định đi tiếp đến retriever hay generator
    def route_after_csv(state: GraphState) -> str:
        """Routing function: sau csv_node, nếu intent=both thì đi đến retriever, ngược lại đi đến generator."""
        intent = state.get("analyzed_intent", "fall_back")  # type: ignore[assignment]
        if intent == "both":
            return "retriever_node"  # Nếu both: đi đến retriever_node
        return "generator"  # Nếu csv_only: đi trực tiếp đến generator

    # Thêm conditional edge từ csv_node: routing dựa trên intent (both -> retriever, csv_only -> generator)
    workflow.add_conditional_edges(
        "csv_node",
        route_after_csv,
        {
            "retriever_node": "retriever_node",
            "generator": "generator",
        },
    )

    # Health path tiếp tục: retriever_node -> doc_grader -> generator -> answer_grader
    # (csv_node có thể đi trực tiếp đến generator nếu csv_only)
    workflow.add_edge("retriever_node", "doc_grader")  # Lọc documents không liên quan
    workflow.add_edge("doc_grader", "generator")  # Sinh câu trả lời từ context đã lọc
    workflow.add_edge("generator", "answer_grader")  # Kiểm tra chất lượng câu trả lời

    # Answer grader decision: quyết định kết thúc hay retry
    def route_from_answer_grader(state: GraphState) -> str:
        """Routing function: quyết định kết thúc (end) hay retry (rewrite) dựa trên answer_valid và retry_count."""
        answer_valid = bool(state.get("answer_valid", True))
        retry_count = int(state.get("retry_count", 0))
        
        # Nếu câu trả lời hợp lệ -> kết thúc
        if answer_valid:
            return "end"
        
        # Nếu đã retry quá số lần cho phép -> kết thúc (tránh vòng lặp vô hạn)
        if retry_count >= retries_limit:
            return "end"
        
        # Nếu câu trả lời không hợp lệ và chưa vượt quá retry limit -> retry
        return "rewrite"

    # Thêm conditional edge từ answer_grader: routing dựa trên answer_valid và retry_count
    workflow.add_conditional_edges(
        "answer_grader",
        route_from_answer_grader,
        {
            "end": END,  # Kết thúc graph
            "rewrite": "rewriter",  # Retry: viết lại query
        },
    )

    # Retry loop: rewriter -> quay lại query_analysis để thử lại
    workflow.add_edge("rewriter", "query_analysis")

    # Compile graph với MemorySaver (checkpoint để lưu chat history giữa các lượt hỏi)
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app



