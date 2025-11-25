"""
Package RAG - Retrieval-Augmented Generation cho AEYE

Package này chứa các module xử lý RAG:
- Vector database: Lưu trữ và truy xuất vector health data
- Embedding: Chuyển đổi health data thành vector
- Retrieval: Tìm kiếm dữ liệu sức khỏe liên quan
- Generation: Tạo câu trả lời health recommendation
"""

from .retrieval_agent import RetrievalAgent
from .recommend_agent import RecommendAgent

__all__ = [
    "RetrievalAgent",
    "RecommendAgent"
]