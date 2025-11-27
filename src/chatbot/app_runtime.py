"""Runtime assembly for the production chatbot app.

Module này khởi tạo:
- LLM Google Gemini (miễn phí theo quota).
- Chroma vector store cho tài liệu y tế.
- Pandas DataFrame agent cho `summary.csv`.
- LangGraph app hoàn chỉnh (build_graph).

Mục tiêu: cung cấp hàm `create_chatbot_app()` để GUI/Desktop App có thể gọi.
"""

from __future__ import annotations

from typing import Any

from src.chatbot.config import CHATBOT_CONFIG
from src.chatbot.graph import build_graph
from src.chatbot.llm_factory import create_production_llm
from src.chatbot.tools.csv_loader import create_summary_agent, load_summary_dataframe
from src.chatbot.tools.vector_store import build_or_load_medical_vector_store


def create_chatbot_app() -> Any:
    """Create the full LangGraph chatbot app using production dependencies.

    Returns
    -------
    Any
        Compiled LangGraph app (có thể dùng .invoke()).
    """
    # Validate config trước khi chạy thật
    CHATBOT_CONFIG.validate()

    # LLM Gemini
    llm = create_production_llm(CHATBOT_CONFIG)

    # Vector store cho tài liệu y tế
    vector_store = build_or_load_medical_vector_store(CHATBOT_CONFIG)

    # CSV / summary DataFrame agent
    df = load_summary_dataframe()
    csv_agent = create_summary_agent(df, llm, verbose=False)

    # Build LangGraph app
    app = build_graph(
        llm=llm,
        vector_store=vector_store,
        csv_agent=csv_agent,
        max_retries=CHATBOT_CONFIG.max_retries,
    )
    return app



