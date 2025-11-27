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
    retries_limit = max_retries if max_retries is not None else CHATBOT_CONFIG.max_retries

    workflow = StateGraph(GraphState)

    # --- Node wrappers với dependency injection ---
    workflow.add_node("guardrails", guardrails_node)
    workflow.add_node("social_bot", partial(social_response_node, llm=llm))
    workflow.add_node("contextualize", partial(contextualize_node, llm=llm))
    workflow.add_node("query_analysis", analyze_query_node)
    workflow.add_node("csv_node", partial(csv_analyst_node, agent=csv_agent))
    workflow.add_node("retriever_node", partial(medical_retriever_node, vector_store=vector_store))
    workflow.add_node("doc_grader", doc_grader_node)
    workflow.add_node("generator", partial(generator_node, llm=llm))
    workflow.add_node("answer_grader", answer_grader_node)
    workflow.add_node("rewriter", partial(rewriter_node, llm=llm))

    # --- Edges ---
    workflow.set_entry_point("guardrails")

    # Guardrails: route social vs health
    def route_from_guardrails(state: GraphState) -> str:
        route = state.get("route", "health")  # type: ignore[assignment]
        if route == "social":
            return "social_bot"
        return "contextualize"

    workflow.add_conditional_edges(
        "guardrails",
        route_from_guardrails,
        {
            "social_bot": "social_bot",
            "contextualize": "contextualize",
        },
    )

    # Social path -> END
    workflow.add_edge("social_bot", END)

    # Health path
    workflow.add_edge("contextualize", "query_analysis")

    # Query analysis routing
    def route_from_query_analysis(state: GraphState) -> str:
        intent = state.get("analyzed_intent", "fall_back")  # type: ignore[assignment]
        if intent == "realtime_data":
            return "csv_only"
        if intent == "chunked_data":
            return "retriever_only"
        if intent == "both":
            return "both"
        # fall_back: chỉ dùng generator với context hiện tại (có thể rỗng)
        return "no_rag"

    workflow.add_conditional_edges(
        "query_analysis",
        route_from_query_analysis,
        {
            "csv_only": "csv_node",
            "retriever_only": "retriever_node",
            "both": "csv_node",  # Sau csv_node ta sẽ nối sang retriever_node
            "no_rag": "generator",
        },
    )

    # CSV node routing: csv-only -> generator, both -> retriever
    def route_after_csv(state: GraphState) -> str:
        intent = state.get("analyzed_intent", "fall_back")  # type: ignore[assignment]
        if intent == "both":
            return "retriever_node"
        return "generator"

    workflow.add_conditional_edges(
        "csv_node",
        route_after_csv,
        {
            "retriever_node": "retriever_node",
            "generator": "generator",
        },
    )

    # Sau khi có context từ CSV hoặc retriever (hoặc cả hai) -> doc_grader -> generator -> answer_grader
    workflow.add_edge("retriever_node", "doc_grader")
    workflow.add_edge("doc_grader", "generator")
    workflow.add_edge("generator", "answer_grader")

    # Answer grader decision
    def route_from_answer_grader(state: GraphState) -> str:
        answer_valid = bool(state.get("answer_valid", True))
        retry_count = int(state.get("retry_count", 0))
        if answer_valid:
            return "end"
        if retry_count >= retries_limit:
            return "end"
        return "rewrite"

    workflow.add_conditional_edges(
        "answer_grader",
        route_from_answer_grader,
        {
            "end": END,
            "rewrite": "rewriter",
        },
    )

    # Rewriter -> quay lại query_analysis
    workflow.add_edge("rewriter", "query_analysis")

    # Compile với MemorySaver (chat history checkpoint)
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app



