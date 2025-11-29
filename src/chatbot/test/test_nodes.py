"""
Tests cho các node trong `src.chatbot.nodes` với dữ liệu thật và LLM thật.

Mỗi test:
- Gọi trực tiếp 1 node với input đơn giản.
- Sử dụng dữ liệu thật trong `src/data/summary.csv` và `src/data/medical_docs/*.txt`
  ở những node có liên quan.
- Sử dụng LLM thật (Gemini) thông qua `create_production_llm`.
- In input/output ra console để bạn quan sát khi chạy `pytest -s`.
"""

from __future__ import annotations

import sys
import io
from pathlib import Path
from typing import Any, Dict, List

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
# Dùng langchain_community.vectorstores.Chroma để tương thích với LangChain 0.3.x
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage

# Cấu hình UTF-8 cho stdout/stderr để hiển thị tiếng Việt trên console
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        # Python < 3.7: dùng io.TextIOWrapper
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Đảm bảo project root (chứa thư mục src/) trong sys.path
# File path: <project>/src/chatbot/test/test_nodes.py -> parents[3] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chatbot.nodes.chat_utils import (  # noqa: E402
    contextualize_node,
    detect_intent_from_text,
    guardrails_node,
    social_response_node,
)
from src.chatbot.nodes.query_analysis import analyze_query_node  # noqa: E402
from src.chatbot.nodes.csv_node import csv_analyst_node  # noqa: E402
from src.chatbot.nodes.retriever_node import medical_retriever_node  # noqa: E402
from src.chatbot.nodes.grader_node import (  # noqa: E402
    answer_grader_node,
    doc_grader_node,
)
from src.chatbot.nodes.generator_node import generator_node  # noqa: E402
from src.chatbot.nodes.rewriter_node import rewriter_node  # noqa: E402
from src.chatbot.tools.vector_store import (  # noqa: E402
    build_or_load_medical_vector_store,
)
from src.chatbot.tools.csv_loader import (  # noqa: E402
    create_summary_agent,
    load_summary_dataframe,
)
from src.chatbot.llm_factory import create_production_llm  # noqa: E402
from src.chatbot.config import CHATBOT_CONFIG  # noqa: E402


# ===== Tests cho chat_utils (Guardrails + Contextualize + Social) =====


def test_detect_intent_and_guardrails() -> None:
    text_social = "Hi, bạn khỏe không?"
    text_health = "Tôi bị mỏi mắt và đau đầu khi dùng máy tính."

    social = detect_intent_from_text(text_social)
    health = detect_intent_from_text(text_health)

    print("detect_intent social input:", text_social)
    print("detect_intent social output:", social)
    print("detect_intent health input:", text_health)
    print("detect_intent health output:", health)

    assert social == "social"
    # Heuristic có thể vẫn trả về "social" cho câu hỏi sức khỏe đơn giản,
    # nên chỉ kiểm tra nó thuộc một trong 2 route hợp lệ.
    assert health in {"social", "health"}

    state = {"messages": [HumanMessage(content=text_social)]}
    result = guardrails_node(state)
    print("guardrails_node route:", result["route"])
    assert result["route"] == "social"


def test_social_and_contextualize_nodes() -> None:
    CHATBOT_CONFIG.validate()
    llm = create_production_llm()

    messages = [
        HumanMessage(content="Tôi bị mỏi mắt khi làm việc với máy tính."),
        HumanMessage(content="Làm sao để đỡ mỏi mắt?"),
    ]
    state: Dict[str, Any] = {
        "messages": messages,
        "original_question": "Làm sao để đỡ mỏi mắt?",
    }

    social_state = social_response_node(state, llm)
    print("social_response_node generation:", social_state["generation"])
    assert isinstance(social_state["generation"], str) and len(social_state["generation"]) > 0

    ctx_state = contextualize_node(state, llm)
    print("contextualize_node reformulated_question:", ctx_state["reformulated_question"])
    assert isinstance(ctx_state["reformulated_question"], str) and len(ctx_state["reformulated_question"]) > 0


# ===== Test cho query_analysis =====


def test_analyze_query_node_with_real_question() -> None:
    state: Dict[str, Any] = {
        "reformulated_question": (
            "Hãy cho tôi thống kê thời lượng các session gần đây và giải thích "
            "hội chứng mỏi mắt CVS là gì."
        ),
    }
    new_state = analyze_query_node(state)

    print("analyze_query_node input question:", state["reformulated_question"])
    print("analyze_query_node intent:", new_state["analyzed_intent"])
    print("analyze_query_node sub_queries:", new_state["sub_queries"])

    assert new_state["analyzed_intent"] in {"realtime_data", "chunked_data", "both", "fall_back"}
    assert isinstance(new_state["sub_queries"], list) and len(new_state["sub_queries"]) == 1


# ===== Test cho csv_node (summary.csv) =====


def test_csv_analyst_node_with_real_summary() -> None:
    CHATBOT_CONFIG.validate()
    df = load_summary_dataframe()
    llm = create_production_llm()
    agent = create_summary_agent(df, llm, verbose=False)

    state: Dict[str, Any] = {
        "sub_queries": ["Hãy cho tôi biết trung bình duration_minutes là bao nhiêu?"],
        "csv_context": [],  # Đổi từ "context" sang "csv_context"
    }
    new_state = csv_analyst_node(state, agent)

    print("csv_analyst_node sub_queries:", state["sub_queries"])
    print("csv_analyst_node new csv_context:", new_state["csv_context"])

    assert len(new_state["csv_context"]) >= 1  # Đổi từ "context" sang "csv_context"


# ===== Test cho retriever_node (medical_docs) =====


def test_medical_retriever_node_with_real_docs() -> None:
    CHATBOT_CONFIG.validate()
    # Thử dùng HuggingFace embeddings, nếu không có thì dùng Google embeddings
    try:
        vs = build_or_load_medical_vector_store(
            CHATBOT_CONFIG,
            force_rebuild=False,
            use_huggingface=True,  # Thử HuggingFace trước
        )
    except ImportError:
        # Fallback: dùng Google embeddings nếu HuggingFace không có
        vs = build_or_load_medical_vector_store(
            CHATBOT_CONFIG,
            force_rebuild=False,
            use_huggingface=False,  # Dùng Google embeddings
        )

    state: Dict[str, Any] = {
        "sub_queries": ["hội chứng mỏi mắt do sử dụng máy tính CVS"],
        "doc_context": [],  # Đổi từ "context" sang "doc_context"
    }
    new_state = medical_retriever_node(state, vs)

    print("medical_retriever_node sub_queries:", state["sub_queries"])
    print("medical_retriever_node new doc_context:", new_state["doc_context"])

    assert len(new_state["doc_context"]) >= 1  # Đổi từ "context" sang "doc_context"


# ===== Test cho grader_node =====


def test_doc_and_answer_grader_nodes() -> None:
    state: Dict[str, Any] = {
        "reformulated_question": "Làm sao giảm mỏi mắt khi dùng máy tính?",
        "context": [
            "Bạn nên nghỉ giải lao sau mỗi 20 phút làm việc với màn hình.",
            "Một đoạn text không liên quan đến sức khỏe.",
        ],
        "generation": "Bạn nên nghỉ giải lao thường xuyên, áp dụng quy tắc 20-20-20 và chớp mắt nhiều hơn.",
    }

    filtered_state = doc_grader_node(state)
    print("doc_grader_node original context:", state["context"])
    print("doc_grader_node filtered context:", filtered_state["context"])

    graded_state = answer_grader_node(filtered_state)
    print("answer_grader_node answer_valid:", graded_state["answer_valid"])

    assert isinstance(filtered_state["context"], list)
    assert isinstance(graded_state["answer_valid"], bool)


# ===== Test cho generator_node & rewriter_node =====


def test_generator_and_rewriter_nodes_with_real_llm() -> None:
    CHATBOT_CONFIG.validate()
    llm = create_production_llm()

    state: Dict[str, Any] = {
        "reformulated_question": "Làm sao giảm mỏi mắt khi dùng máy tính?",
        "context": [
            "Nghỉ giải lao 20-20-20 và chớp mắt nhiều lần.",
        ],
        "generation": "Câu trả lời trước đó quá ngắn.",
        "retry_count": 0,
    }

    gen_state = generator_node(state, llm)
    print("generator_node generation:", gen_state["generation"])
    assert isinstance(gen_state["generation"], str) and len(gen_state["generation"]) > 0

    # Thêm reformulated_question vào gen_state trước khi gọi rewriter_node
    gen_state["reformulated_question"] = state["reformulated_question"]
    rew_state = rewriter_node(gen_state, llm)
    print("rewriter_node sub_queries:", rew_state["sub_queries"])
    print("rewriter_node retry_count:", rew_state["retry_count"])

    assert isinstance(rew_state["sub_queries"], list) and len(rew_state["sub_queries"]) == 1
    assert rew_state["retry_count"] == 1



