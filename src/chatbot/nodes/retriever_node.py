"""Medical retriever node: truy vấn vector store Chroma."""

from __future__ import annotations

from typing import Dict, List, MutableMapping

# Dùng langchain_community.vectorstores.Chroma để tương thích với LangChain 0.3.x
from langchain_community.vectorstores import Chroma


StateDict = MutableMapping[str, object]


def medical_retriever_node(state: StateDict, vector_store: Chroma) -> Dict[str, object]:
    """Retrieve medical docs cho từng sub_query và tạo doc_context mới.
    
    QUAN TRỌNG: Node này RESET doc_context (không append vào context cũ) để tránh tích lũy
    context qua nhiều lượt hỏi. Mỗi lần chạy, doc_context được tạo mới từ đầu.
    
    Parameters
    ----------
    state:
        GraphState với field "sub_queries" chứa danh sách câu hỏi cần retrieve.
    vector_store:
        Chroma vector store chứa medical documents.
    
    Returns
    -------
    Dict[str, object]
        {"doc_context": List[str]} - danh sách documents đã retrieve (RESET, không append).
    
    Raises
    ------
    ValueError
        Nếu sub_queries rỗng.
    """
    # Lấy danh sách sub_queries từ state
    sub_queries: List[str] = state.get("sub_queries", [])  # type: ignore[assignment]
    if not sub_queries:
        raise ValueError("medical_retriever_node requires non-empty 'sub_queries'.")

    # RESET doc_context: tạo list mới thay vì append vào context cũ
    doc_context: List[str] = []
    
    # Tạo retriever với k=3 (lấy top 3 documents liên quan nhất)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Retrieve documents cho từng sub_query
    for q in sub_queries:
        # Gọi retriever để tìm documents liên quan
        docs = retriever.invoke(q)
        
        # Nối nội dung của các documents thành một chuỗi
        joined = "\n".join(d.page_content for d in docs)
        doc_context.append(joined)

    # Trả về doc_context mới (RESET, không append vào state cũ)
    return {"doc_context": doc_context}


