"""Medical retriever node: truy vấn vector store Chroma."""

from __future__ import annotations

from typing import Dict, List, MutableMapping

from langchain_community.vectorstores import Chroma


StateDict = MutableMapping[str, object]


def medical_retriever_node(state: StateDict, vector_store: Chroma) -> Dict[str, object]:
    """Retrieve medical docs cho từng sub_query và append vào context."""
    sub_queries: List[str] = state.get("sub_queries", [])  # type: ignore[assignment]
    if not sub_queries:
        raise ValueError("medical_retriever_node requires non-empty 'sub_queries'.")

    doc_context: List[str] = list(state.get("doc_context", []))  # type: ignore[assignment]
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    for q in sub_queries:
        docs = retriever.invoke(q)
        joined = "\n".join(d.page_content for d in docs)
        doc_context.append(joined)

    new_state: Dict[str, object] = dict(state)
    new_state["doc_context"] = doc_context
    return new_state


