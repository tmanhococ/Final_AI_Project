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
    """Run CSV Pandas agent trên các sub_queries và tạo csv_context mới.
    
    QUAN TRỌNG: Node này RESET csv_context (không append vào context cũ) để tránh tích lũy
    context qua nhiều lượt hỏi. Mỗi lần chạy, csv_context được tạo mới từ đầu.
    
    Parameters
    ----------
    state:
        GraphState với field "sub_queries" chứa danh sách câu hỏi cần phân tích.
    agent:
        Pandas DataFrame Agent (có method .invoke()) để truy vấn CSV.
    
    Returns
    -------
    Dict[str, object]
        {"csv_context": List[str]} - danh sách kết quả phân tích CSV (RESET, không append).
    
    Raises
    ------
    ValueError
        Nếu sub_queries rỗng.
    """
    # Lấy danh sách sub_queries từ state
    sub_queries: List[str] = state.get("sub_queries", [])  # type: ignore[assignment]
    if not sub_queries:
        raise ValueError("csv_analyst_node requires non-empty 'sub_queries'.")

    # RESET csv_context: tạo list mới thay vì append vào context cũ
    csv_context: List[str] = []

    # Xử lý từng sub_query
    for q in sub_queries:
        try:
            # Gọi agent để phân tích câu hỏi trên DataFrame
            # Dùng API .invoke mới thay vì run (run đã deprecated)
            answer = agent.invoke({"input": q})  # type: ignore[arg-type]
            
            # Agent có thể trả về dict hoặc str tùy implementation
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
            # Xử lý các lỗi khác (network, API quota, etc.)
            csv_context.append(f"Lỗi khi truy vấn dữ liệu: {str(e)}")

    # Trả về csv_context mới (RESET, không append vào state cũ)
    return {"csv_context": csv_context}


