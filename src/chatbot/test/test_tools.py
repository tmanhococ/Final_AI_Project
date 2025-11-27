"""
Tests for tools: vector_store and csv_loader với dữ liệu thật.

Các test này:
- Dùng embedding thật (hoặc SafeGoogleEmbeddings với fallback) để build Chroma DB
  từ `src/data/medical_docs/*.txt` và lưu vào `data/chroma_db`.
- Dùng LLM thật (Gemini) để tạo Pandas DataFrame Agent trên `src/data/summary.csv`.
- In ra input/output để bạn dễ kiểm tra khi chạy pytest -s.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import pytest
from langchain_core.documents import Document

# Đảm bảo thư mục chứa package `src` nằm trong sys.path
# Path hiện tại: <project>/src/chatbot/test/test_*.py
# => parents[3] = thư mục `src`
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chatbot.tools.vector_store import build_chroma_from_documents  # noqa: E402
from src.chatbot.tools.csv_loader import create_summary_agent, load_summary_dataframe  # noqa: E402
from src.chatbot.llm_factory import create_production_llm  # noqa: E402
from src.chatbot.config import CHATBOT_CONFIG  # noqa: E402


# ===== Tests cho vector_store =====


def test_build_chroma_from_documents_basic(tmp_path) -> None:
    """Build REAL Chroma DB from medical_docs và test truy vấn đơn giản."""
    # Đảm bảo config hợp lệ (có GOOGLE_API_KEY nếu còn quota)
    CHATBOT_CONFIG.validate()

    # Build hoặc load vector store thật (sử dụng SafeGoogleEmbeddings bên trong)
    from src.chatbot.tools.vector_store import build_or_load_medical_vector_store  # noqa: E402

    vs = build_or_load_medical_vector_store(CHATBOT_CONFIG, force_rebuild=True)

    query = "Tôi bị mỏi mắt khi làm việc với máy tính, hội chứng CVS là gì?"
    retriever = vs.as_retriever(search_kwargs={"k": 3})
    retrieved = retriever.invoke(query)

    print("Vector store query input (real docs):", query)
    print("Vector store retrieved docs (real docs):")
    for i, d in enumerate(retrieved, start=1):
        print(f"  Doc {i}:", d.page_content)

    assert len(retrieved) > 0


# ===== Tests cho csv_loader (Pandas DataFrame Agent) =====


@pytest.mark.parametrize("use_real_csv", [True, False])
def test_create_summary_agent_and_run(tmp_path, use_real_csv: bool) -> None:
    """Kiểm tra tạo Pandas DataFrame Agent trên summary.csv (hoặc mock).

    - Nếu ``use_real_csv=True`` và file tồn tại: dùng dữ liệu thật.
    - Nếu không: tạo DataFrame nhỏ mock để test logic agent.
    """
    if use_real_csv:
        # Dùng đường dẫn mặc định trong csv_loader (src/data/summary.csv)
        df = load_summary_dataframe()
        source_desc = "real CSV at src/data/summary.csv"
    else:
        # Mock DataFrame nhỏ cho test logic
        df = pd.DataFrame(
            {
                "session_id": ["s1", "s2"],
                "duration_minutes": [0.5, 1.2],
                "avg_ear": [0.25, 0.30],
                "drowsiness_events": [0, 1],
            }
        )
        source_desc = "mock DataFrame"

    print("Summary DataFrame source:", source_desc)
    print("Summary DataFrame head:\n", df.head())

    # Dùng LLM thật qua factory (sẽ sử dụng GOOGLE_API_KEY từ .env)
    llm = create_production_llm()
    agent = create_summary_agent(df, llm, verbose=False)

    question = "Hãy cho tôi biết trung bình thời lượng các phiên đo (duration_minutes) là bao nhiêu phút?"
    print("Summary DataFrame source:", source_desc)
    print("Summary DataFrame head:\n", df.head())

    # Truy vấn thực tế agent (LLM thật) – có thể tốn quota LLM nhưng bạn yêu cầu dùng thật
    result = agent.invoke({"input": question})
    # AgentExecutor thường trả về dict với key 'output'
    answer_text = result.get("output", str(result)) if isinstance(result, dict) else str(result)

    print("Pandas agent question:", question)
    print("Pandas agent answer:", answer_text)

    assert isinstance(answer_text, str) and len(answer_text) > 0


