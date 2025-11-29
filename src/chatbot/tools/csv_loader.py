"""CSV loader and Pandas DataFrame agent utilities for health summaries.

Module này tập trung xử lý:
- Đọc file ``summary.csv`` (log tóm tắt các phiên đo sức khỏe) thành ``pd.DataFrame``.
- Tạo Pandas DataFrame Agent với hướng dẫn chi tiết về ý nghĩa các trường dữ liệu.

Core logic được thiết kế thuần (input là DataFrame, LLM) để dễ kiểm thử.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from langchain_core.language_models import BaseLanguageModel

from src.chatbot.config import CHATBOT_CONFIG, ChatbotConfig


PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ============================================
# FIELD DESCRIPTIONS FOR PANDAS AGENT
# ============================================
CSV_FIELD_DESCRIPTIONS = """
## Mô tả các trường dữ liệu trong DataFrame 'df':

### 1. session_id (str)
- Định dạng: YYYYMMDD_HHMMSS (ví dụ: 20251126_145533)
- Ý nghĩa: Mã định danh duy nhất của mỗi phiên đo sức khỏe

### 2. start_time (str/datetime)
- Định dạng: ISO 8601 (ví dụ: 2025-11-26T14:55:33.978605)
- Ý nghĩa: Thời điểm BẮT ĐẦU phiên đo

### 3. end_time (str/datetime)
- Định dạng: ISO 8601 (ví dụ: 2025-11-26T14:55:58.174940)
- Ý nghĩa: Thời điểm KẾT THÚC phiên đo

### 4. duration_minutes (float)
- Đơn vị: PHÚT
- Ý nghĩa: THỜI LƯỢNG của phiên đo (end_time - start_time)
- Từ khóa người dùng: "bao lâu", "thời gian", "duration", "phút", "lâu", "thời lượng"
- Ví dụ: 0.4 phút = 24 giây, 0.64 phút = 38 giây

### 5. avg_ear (float)
- Tên đầy đủ: Eye Aspect Ratio (tỷ lệ khung hình mắt)
- Giá trị: 0.0 - 1.0 (giá trị càng THẤP = mắt càng NHẮM)
- Ý nghĩa: Mức độ MỞ/NHẮM MẮT trung bình trong phiên
- Từ khóa: "mắt", "eye", "ear", "mỏi mắt", "nhắm mắt", "chớp mắt", "mở mắt"
- Ngưỡng: 
  * EAR < 0.25: Mắt đang NHẮM hoặc rất MỎI
  * EAR 0.25-0.35: Mắt MỞ BÌNH THƯỜNG
  * EAR > 0.35: Mắt MỞ TO (tỉnh táo)

### 6. avg_distance_cm (float)
- Đơn vị: CENTIMET (cm)
- Ý nghĩa: Khoảng cách TRUNG BÌNH từ MẶT đến MÀN HÌNH/CAMERA
- Từ khóa: "khoảng cách", "distance", "gần", "xa", "cách màn hình", "cm"
- Ngưỡng khuyến nghị:
  * < 40cm: QUÁ GẦN màn hình (nguy cơ mỏi mắt cao)
  * 40-70cm: Khoảng cách AN TOÀN
  * > 70cm: Xa màn hình (có thể ngồi sai tư thế)

### 7. drowsiness_events (int)
- Đơn vị: SỐ LẦN
- Ý nghĩa: Số lần phát hiện BUỒN NGỦ/MỎI MẮT trong phiên
- Từ khóa: "buồn ngủ", "drowsiness", "ngủ gật", "mệt", "sự kiện", "cảnh báo"
- Ngưỡng:
  * 0: KHÔNG có dấu hiệu buồn ngủ
  * 1-3: Có dấu hiệu mệt mỏi NHẸ
  * > 3: Mệt mỏi NGHIÊM TRỌNG

### 8. avg_shoulder_tilt (float)
- Đơn vị: ĐỘ (degrees)
- Ý nghĩa: Góc NGHIÊNG VAI trung bình (lệch trái/phải)
- Từ khóa: "vai", "shoulder", "nghiêng", "tilt", "cong lưng", "tư thế vai"
- Ngưỡng:
  * 0-5°: Vai THẲNG (tư thế tốt)
  * 5-10°: Vai hơi lệch (chấp nhận được)
  * > 10°: Vai NGHIÊNG NHIỀU (tư thế xấu)

### 9. avg_head_pitch (float)
- Đơn vị: ĐỘ (degrees)
- Ý nghĩa: Góc NGỬA/CÚI ĐẦU trung bình (pitch = xoay quanh trục ngang)
- Từ khóa: "đầu", "head", "pitch", "cúi", "ngửa", "gật đầu", "tư thế đầu"
- Ngưỡng:
  * -10° đến +10°: Đầu THẲNG (tư thế tốt)
  * +10° đến +20°: Đầu CÚI NHẸ (nhìn xuống màn hình)
  * > +20°: Đầu CÚI NHIỀU (tư thế xấu, gây đau cổ)
  * < -10°: Đầu NGỬA (hiếm gặp)

### 10. avg_head_yaw (float)
- Đơn vị: ĐỘ (degrees)
- Ý nghĩa: Góc QUAY TRÁI/PHẢI ĐẦU trung bình (yaw = xoay quanh trục dọc)
- Từ khóa: "đầu", "head", "yaw", "quay", "nhìn sang", "lắc đầu", "hướng nhìn"
- Ngưỡng:
  * -5° đến +5°: Đầu THẲNG (nhìn thẳng màn hình)
  * ±5° đến ±15°: Đầu QUAY NHẸ
  * > ±15°: Đầu QUAY NHIỀU (không tập trung vào màn hình)

## Hướng dẫn phân tích:
- Khi người dùng hỏi về "tư thế", "posture": phân tích shoulder_tilt, head_pitch, head_yaw
- Khi hỏi về "mỏi mắt", "sức khỏe mắt": phân tích avg_ear, avg_distance_cm, drowsiness_events
- Khi hỏi về "thời gian": sử dụng duration_minutes hoặc start_time/end_time
- Khi hỏi về "ngày nào", "thời điểm": parse session_id hoặc start_time
- Luôn trả lời bằng TIẾNG VIỆT, dễ hiểu, có đơn vị đo
"""


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

    Agent được cấu hình với:
    - Mô tả chi tiết về ý nghĩa các trường dữ liệu
    - Hướng dẫn phân tích và từ khóa người dùng
    - Prefix để agent hiểu ngữ cảnh y tế/sức khỏe

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
        LangChain agent that can be used with ``agent.invoke({"input": question})``.
    """
    try:
        from langchain_experimental.agents.agent_toolkits import (  # type: ignore
            create_pandas_dataframe_agent,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "langchain_experimental chưa được cài. "
            "Hãy chạy: pip install langchain-experimental"
        ) from exc

    # Tạo prefix instruction cho agent
    prefix = f"""
Bạn là trợ lý phân tích dữ liệu sức khỏe. Bạn đang làm việc với DataFrame 'df' chứa dữ liệu các phiên đo sức khỏe qua camera.

{CSV_FIELD_DESCRIPTIONS}

Khi trả lời:
1. Sử dụng pandas để phân tích dữ liệu (df.describe(), df.mean(), df.groupby(), v.v.)
2. Trả lời bằng TIẾNG VIỆT, dễ hiểu
3. Bao gồm ĐƠN VỊ ĐO (phút, cm, độ) trong câu trả lời
4. Nếu có giá trị bất thường (outlier), CHỈ RA và GIẢI THÍCH
5. So sánh với NGƯỠNG KHUYẾN NGHỊ để đánh giá tình trạng sức khỏe

Ví dụ câu trả lời tốt:
"Thời lượng trung bình của các phiên là 0.45 phút (27 giây). Khoảng cách trung bình là 57.74 cm, nằm trong ngưỡng an toàn (40-70cm)."
"""

    # Tạo agent với prefix instruction
    # Không chỉ định agent_type để LangChain tự động chọn phù hợp với LLM
    # (Gemini không hỗ trợ "openai-tools", cần dùng "zero-shot-react-description")
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        prefix=prefix,
        verbose=verbose,
        allow_dangerous_code=True,  # Cần cho môi trường local/học tập
        handle_parsing_errors=True,  # Xử lý lỗi parsing từ LLM (quan trọng cho Gemini)
    )
    return agent