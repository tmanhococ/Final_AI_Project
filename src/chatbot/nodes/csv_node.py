"""CSV Analyst node: truy vấn dữ liệu log qua Pandas DataFrame Agent.

Node này nhận:
- state["sub_queries"]: List[str]
- Một agent đã được tạo sẵn (AgentExecutor) trên DataFrame summary/log.

Và trả về:
- state["context"]: append thêm chuỗi mô tả kết quả phân tích.
"""

from __future__ import annotations

from typing import Dict, List, MutableMapping

from typing import Protocol, Any


class SupportsInvoke(Protocol):
    """Small protocol để chấp nhận mọi object có .invoke(dict) -> Any."""

    def invoke(self, input: Dict[str, Any], config: Any | None = None, **kwargs: Any) -> Any:  # pragma: no cover - protocol
        ...


StateDict = MutableMapping[str, object]


def csv_analyst_node(state: StateDict, agent: SupportsInvoke) -> Dict[str, object]:
    """Run CSV Pandas agent trên các sub_queries và append vào context."""
    sub_queries: List[str] = state.get("sub_queries", [])  # type: ignore[assignment]
    if not sub_queries:
        raise ValueError("csv_analyst_node requires non-empty 'sub_queries'.")

    csv_context: List[str] = list(state.get("csv_context", []))  # type: ignore[assignment]

    for q in sub_queries:
        try:
            # Dùng API .invoke mới thay vì run (run đã deprecated).
            answer = agent.invoke({"input": q})  # type: ignore[arg-type]
            # answer có thể là dict hoặc str tùy agent; ép về str để append.
            if isinstance(answer, dict):
                # Lấy output từ dict nếu có
                output = answer.get("output", str(answer))
                csv_context.append(str(output))
            else:
                csv_context.append(str(answer))
        except ValueError as e:
            # Xử lý lỗi parsing: Gemini thường trả lời đúng nội dung nhưng sai format ReAct
            error_msg = str(e)
            if "Could not parse LLM output:" in error_msg:
                # Trích xuất phần text mô hình đã trả lời từ thông báo lỗi
                try:
                    # Format: "Could not parse LLM output: `...`"
                    parts = error_msg.split("Could not parse LLM output: `")
                    if len(parts) > 1:
                        raw_response = parts[1].split("`")[0]
                        csv_context.append(raw_response)
                    else:
                        # Fallback: dùng toàn bộ error message
                        csv_context.append(f"Kết quả phân tích: {error_msg}")
                except Exception:
                    csv_context.append(f"Kết quả phân tích: {error_msg}")
            else:
                csv_context.append(f"Lỗi phân tích dữ liệu: {str(e)}")
        except Exception as e:
            # Xử lý các lỗi khác
            csv_context.append(f"Lỗi khi truy vấn dữ liệu: {str(e)}")

    return {"csv_context": csv_context}


