"""CSV loader and Pandas DataFrame agent utilities for health summaries.

Module này tập trung xử lý:
- Đọc file ``summary.csv`` (log tóm tắt các phiên đo sức khỏe) thành ``pd.DataFrame``.
- Tạo Pandas DataFrame Agent để trả lời câu hỏi phân tích thống kê đơn giản.

Core logic được thiết kế thuần (input là DataFrame, LLM) để dễ kiểm thử.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from langchain_core.language_models import BaseLanguageModel

from src.chatbot.config import CHATBOT_CONFIG, ChatbotConfig


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def get_summary_csv_path(config: ChatbotConfig = CHATBOT_CONFIG) -> Path:
    """Return default path to ``summary.csv`` used by the vision system.

    Parameters
    ----------
    config:
        Currently unused, but kept for symmetry and future extension.

    Returns
    -------
    Path
        Path to ``summary.csv`` under ``src/data/``.
    """
    # File hiện tại nằm ở: src/data/summary.csv (theo cấu trúc dự án)
    return PROJECT_ROOT / "src" / "data" / "summary.csv"


def load_summary_dataframe(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Load the health summary CSV into a pandas DataFrame.

    Parameters
    ----------
    csv_path:
        Optional custom path. Nếu ``None`` sẽ dùng ``get_summary_csv_path()``.

    Returns
    -------
    pandas.DataFrame
        Loaded DataFrame.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If the CSV is empty or has no rows of data.
    """
    if csv_path is None:
        csv_path = get_summary_csv_path()

    if not csv_path.exists():
        raise FileNotFoundError(f"summary.csv not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty or len(df.columns) == 0:
        raise ValueError(f"summary.csv at {csv_path} is empty or invalid.")

    return df


def create_summary_agent(
    df: pd.DataFrame,
    llm: BaseLanguageModel,
    *,
    verbose: bool = False,
):
    """Create a Pandas DataFrame Agent over the summary DataFrame.

    Parameters
    ----------
    df:
        Health summary DataFrame.
    llm:
        Language model used by the agent (có thể là Dummy LLM trong pytest).
    verbose:
        If ``True``, agent will print intermediate steps.

    Returns
    -------
    AgentExecutor
        LangChain agent that can be used with ``agent.run(question)``.
    """
    try:
        from langchain_experimental.agents.agent_toolkits import (  # type: ignore
            create_pandas_dataframe_agent,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover - môi trường thiếu optional dep
        raise ModuleNotFoundError(
            "langchain_experimental chưa được cài. "
            "Hãy chạy: pip install langchain-experimental"
        ) from exc

    # handle_parsing_errors giúp agent bền hơn khi LLM trả ra format lạ trong quá trình dev.
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=verbose,
        # Bật explicit theo khuyến cáo bảo mật; chỉ dùng trong môi trường local/môn học.
        allow_dangerous_code=True,
    )
    return agent


