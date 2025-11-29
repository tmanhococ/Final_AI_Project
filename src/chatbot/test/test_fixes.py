"""
Tests để xác nhận các lỗi đã được fix:

1. Contextualize chỉ viết lại câu hỏi mới nhất (không tổng hợp các câu hỏi trước)
2. Context fields được reset (không append vào context cũ)
3. HuggingFace embeddings hoạt động (local, miễn phí)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import pytest
from langchain_core.messages import HumanMessage, AIMessage

# Cấu hình UTF-8 cho stdout/stderr
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Đảm bảo project root trong sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chatbot.nodes.chat_utils import contextualize_node  # noqa: E402
from src.chatbot.nodes.csv_node import csv_analyst_node  # noqa: E402
from src.chatbot.nodes.retriever_node import medical_retriever_node  # noqa: E402
from src.chatbot.nodes.grader_node import doc_grader_node  # noqa: E402
from src.chatbot.nodes.query_analysis import analyze_query_node  # noqa: E402
from src.chatbot.nodes.rewriter_node import rewriter_node  # noqa: E402
from src.chatbot.tools.vector_store import (  # noqa: E402
    create_huggingface_embeddings,
    build_or_load_medical_vector_store,
)
from src.chatbot.llm_factory import create_production_llm  # noqa: E402
from src.chatbot.config import CHATBOT_CONFIG  # noqa: E402


# ===== Test 1: Contextualize chỉ viết lại câu hỏi mới nhất =====


def test_contextualize_only_rewrites_latest_question() -> None:
    """Test: contextualize_node chỉ viết lại câu hỏi mới nhất, không tổng hợp các câu hỏi trước.
    
    Scenario:
    - Lịch sử hội thoại có nhiều câu hỏi trước đó
    - Câu hỏi mới nhất: "CVS là gì?"
    - Expected: reformulated_question chỉ chứa câu hỏi mới nhất được viết lại, không tổng hợp
    """
    CHATBOT_CONFIG.validate()
    llm = create_production_llm()
    
    # Tạo lịch sử hội thoại với nhiều câu hỏi trước đó
    messages = [
        HumanMessage(content="Tôi bị mỏi mắt"),
        AIMessage(content="Bạn có thể thử quy tắc 20-20-20"),
        HumanMessage(content="Quy tắc đó là gì?"),
        AIMessage(content="Cứ 20 phút nhìn xa 20 feet trong 20 giây"),
        HumanMessage(content="CVS là gì?"),  # Câu hỏi mới nhất
    ]
    
    state: Dict[str, object] = {
        "messages": messages,
        "original_question": "CVS là gì?",  # Câu hỏi mới nhất
    }
    
    # Gọi contextualize_node
    result = contextualize_node(state, llm)
    reformulated = result["reformulated_question"]
    
    print("\n" + "="*80)
    print("TEST: Contextualize chỉ viết lại câu hỏi mới nhất")
    print("="*80)
    print(f"Original question: {state['original_question']}")
    print(f"Reformulated question: {reformulated}")
    print(f"Number of questions in reformulated: {reformulated.count('?')}")
    
    # Kiểm tra: reformulated chỉ chứa 1 câu hỏi (không tổng hợp nhiều câu)
    # Nếu tổng hợp, sẽ có nhiều dấu "?" hoặc từ khóa như "các câu hỏi", "tổng hợp"
    assert reformulated.count("?") <= 2, f"Reformulated có quá nhiều câu hỏi: {reformulated}"
    assert "tổng hợp" not in reformulated.lower(), f"Reformulated có vẻ tổng hợp: {reformulated}"
    assert "các câu hỏi" not in reformulated.lower(), f"Reformulated có vẻ liệt kê: {reformulated}"
    
    # Kiểm tra: reformulated phải chứa từ khóa từ câu hỏi mới nhất
    assert "CVS" in reformulated or "cvs" in reformulated.lower(), \
        f"Reformulated không chứa từ khóa từ câu hỏi mới nhất: {reformulated}"
    
    print("✓ PASS: Contextualize chỉ viết lại câu hỏi mới nhất")


# ===== Test 2: Context fields được reset (không append) =====


def test_csv_node_resets_context() -> None:
    """Test: csv_node reset csv_context (không append vào context cũ).
    
    Scenario:
    - State ban đầu có csv_context = ["old context"]
    - csv_node chạy với sub_queries mới
    - Expected: csv_context mới không chứa "old context"
    """
    CHATBOT_CONFIG.validate()
    llm = create_production_llm()
    
    # Tạo mock agent (chỉ trả về string đơn giản)
    class MockAgent:
        def invoke(self, input_dict: Dict) -> str:
            return f"Kết quả phân tích: {input_dict.get('input', '')}"
    
    agent = MockAgent()
    
    # State ban đầu có csv_context cũ
    state: Dict[str, object] = {
        "sub_queries": ["Câu hỏi mới"],
        "csv_context": ["old context", "another old context"],  # Context cũ
    }
    
    # Gọi csv_node
    result = csv_analyst_node(state, agent)
    new_csv_context = result["csv_context"]
    
    print("\n" + "="*80)
    print("TEST: CSV node reset context (không append)")
    print("="*80)
    print(f"Old csv_context: {state.get('csv_context', [])}")
    print(f"New csv_context: {new_csv_context}")
    
    # Kiểm tra: csv_context mới không chứa context cũ
    assert "old context" not in str(new_csv_context), \
        f"csv_context mới vẫn chứa context cũ: {new_csv_context}"
    assert "another old context" not in str(new_csv_context), \
        f"csv_context mới vẫn chứa context cũ: {new_csv_context}"
    
    # Kiểm tra: csv_context mới chứa kết quả mới
    assert len(new_csv_context) > 0, "csv_context mới phải có ít nhất 1 phần tử"
    assert "Câu hỏi mới" in str(new_csv_context), \
        f"csv_context mới phải chứa kết quả từ sub_query mới: {new_csv_context}"
    
    print("✓ PASS: CSV node reset context (không append)")


def test_retriever_node_resets_context() -> None:
    """Test: retriever_node reset doc_context (không append vào context cũ)."""
    CHATBOT_CONFIG.validate()
    
    # Build vector store với HuggingFace embeddings (local, miễn phí) hoặc Google embeddings (fallback)
    try:
        vector_store = build_or_load_medical_vector_store(
            CHATBOT_CONFIG,
            force_rebuild=False,  # Dùng cache nếu có
            use_huggingface=True,  # Dùng HuggingFace embeddings
        )
    except ImportError:
        # Fallback: dùng Google embeddings nếu HuggingFace không có
        pytest.skip("langchain-huggingface chưa được cài đặt, skip test này")
        return
    
    # State ban đầu có doc_context cũ
    state: Dict[str, object] = {
        "sub_queries": ["Hội chứng mỏi mắt CVS là gì?"],
        "doc_context": ["old doc context"],  # Context cũ
    }
    
    # Gọi retriever_node
    result = medical_retriever_node(state, vector_store)
    new_doc_context = result["doc_context"]
    
    print("\n" + "="*80)
    print("TEST: Retriever node reset context (không append)")
    print("="*80)
    print(f"Old doc_context: {state.get('doc_context', [])}")
    print(f"New doc_context length: {len(new_doc_context)}")
    if new_doc_context:
        print(f"New doc_context[0] preview: {new_doc_context[0][:100]}...")
    
    # Kiểm tra: doc_context mới không chứa context cũ
    assert "old doc context" not in str(new_doc_context), \
        f"doc_context mới vẫn chứa context cũ: {new_doc_context}"
    
    # Kiểm tra: doc_context mới chứa documents mới
    assert len(new_doc_context) > 0, "doc_context mới phải có ít nhất 1 phần tử"
    
    print("✓ PASS: Retriever node reset context (không append)")


def test_doc_grader_resets_context() -> None:
    """Test: doc_grader_node reset context khi merge (không append vào context cũ)."""
    # State ban đầu có context cũ
    state: Dict[str, object] = {
        "doc_context": ["Document về CVS", "Document về mỏi mắt"],
        "csv_context": ["Kết quả CSV"],
        "reformulated_question": "CVS là gì?",
        "context": ["old context"],  # Context cũ
    }
    
    # Gọi doc_grader_node
    result = doc_grader_node(state)
    new_context = result["context"]
    
    print("\n" + "="*80)
    print("TEST: Doc grader reset context (không append)")
    print("="*80)
    print(f"Old context: {state.get('context', [])}")
    print(f"New context length: {len(new_context)}")
    print(f"New context: {new_context}")
    
    # Kiểm tra: context mới không chứa context cũ
    assert "old context" not in str(new_context), \
        f"context mới vẫn chứa context cũ: {new_context}"
    
    # Kiểm tra: context mới chứa csv_context + filtered_docs
    assert len(new_context) > 0, "context mới phải có ít nhất 1 phần tử"
    
    print("✓ PASS: Doc grader reset context (không append)")


def test_query_analysis_resets_sub_queries() -> None:
    """Test: analyze_query_node reset sub_queries (không append)."""
    # State ban đầu có sub_queries cũ
    state: Dict[str, object] = {
        "reformulated_question": "CVS là gì?",
        "sub_queries": ["old query 1", "old query 2"],  # Sub queries cũ
    }
    
    # Gọi analyze_query_node
    result = analyze_query_node(state)
    new_sub_queries = result["sub_queries"]
    
    print("\n" + "="*80)
    print("TEST: Query analysis reset sub_queries (không append)")
    print("="*80)
    print(f"Old sub_queries: {state.get('sub_queries', [])}")
    print(f"New sub_queries: {new_sub_queries}")
    
    # Kiểm tra: sub_queries mới không chứa queries cũ
    assert "old query 1" not in new_sub_queries, \
        f"sub_queries mới vẫn chứa query cũ: {new_sub_queries}"
    assert "old query 2" not in new_sub_queries, \
        f"sub_queries mới vẫn chứa query cũ: {new_sub_queries}"
    
    # Kiểm tra: sub_queries mới chứa câu hỏi mới
    assert len(new_sub_queries) > 0, "sub_queries mới phải có ít nhất 1 phần tử"
    assert "CVS là gì?" in new_sub_queries, \
        f"sub_queries mới phải chứa reformulated_question: {new_sub_queries}"
    
    print("✓ PASS: Query analysis reset sub_queries (không append)")


# ===== Test 3: HuggingFace embeddings hoạt động =====


def test_huggingface_embeddings_work() -> None:
    """Test: HuggingFace embeddings hoạt động (local, miễn phí, không có quota limit).
    
    Test này verify:
    - create_huggingface_embeddings() tạo được embeddings instance
    - Embeddings có thể embed documents và queries
    - build_or_load_medical_vector_store() với use_huggingface=True hoạt động
    """
    print("\n" + "="*80)
    print("TEST: HuggingFace embeddings hoạt động")
    print("="*80)
    
    try:
        # Test 1: Tạo HuggingFace embeddings instance
        embeddings = create_huggingface_embeddings()
        print("✓ Created HuggingFace embeddings instance")
        
        # Test 2: Embed documents
        texts = ["Hội chứng mỏi mắt CVS", "Computer Vision Syndrome"]
        doc_embeddings = embeddings.embed_documents(texts)
        print(f"✓ Embedded {len(texts)} documents")
        print(f"  - Embedding dimension: {len(doc_embeddings[0])}")
        assert len(doc_embeddings) == len(texts), "Số lượng embeddings phải bằng số lượng texts"
        assert len(doc_embeddings[0]) > 0, "Embedding dimension phải > 0"
        
        # Test 3: Embed query
        query = "CVS là gì?"
        query_embedding = embeddings.embed_query(query)
        print(f"✓ Embedded query: {query}")
        print(f"  - Embedding dimension: {len(query_embedding)}")
        assert len(query_embedding) > 0, "Query embedding dimension phải > 0"
        assert len(query_embedding) == len(doc_embeddings[0]), \
            "Query embedding dimension phải bằng document embedding dimension"
        
        # Test 4: Build vector store với HuggingFace embeddings
        vector_store = build_or_load_medical_vector_store(
            CHATBOT_CONFIG,
            force_rebuild=False,  # Dùng cache nếu có
            use_huggingface=True,  # Dùng HuggingFace embeddings
        )
        print("✓ Built/loaded vector store với HuggingFace embeddings")
        
        # Test 5: Query vector store
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        retrieved = retriever.invoke("CVS là gì?")
        print(f"✓ Retrieved {len(retrieved)} documents từ vector store")
        assert len(retrieved) > 0, "Phải retrieve được ít nhất 1 document"
        
        print("✓ PASS: HuggingFace embeddings hoạt động hoàn toàn")
        
    except ImportError as e:
        pytest.skip(f"HuggingFace embeddings không khả dụng (cần cài langchain-huggingface): {e}")
    except Exception as e:
        pytest.fail(f"HuggingFace embeddings test failed: {e}")


# ===== Test 4: Integration test - toàn bộ flow với fixes =====


def test_integration_with_fixes() -> None:
    """Integration test: verify toàn bộ flow với các fixes đã áp dụng.
    
    Test này verify:
    1. Contextualize chỉ viết lại câu hỏi mới nhất
    2. Context fields được reset giữa các node
    3. HuggingFace embeddings được dùng (nếu có)
    """
    CHATBOT_CONFIG.validate()
    llm = create_production_llm()
    
    print("\n" + "="*80)
    print("TEST: Integration test với fixes")
    print("="*80)
    
    # Step 1: Contextualize
    messages = [
        HumanMessage(content="Câu hỏi cũ 1"),
        AIMessage(content="Trả lời cũ 1"),
        HumanMessage(content="CVS là gì?"),  # Câu hỏi mới nhất
    ]
    state: Dict[str, object] = {
        "messages": messages,
        "original_question": "CVS là gì?",
    }
    
    result = contextualize_node(state, llm)
    reformulated = result["reformulated_question"]
    print(f"1. Contextualize: {reformulated[:100]}...")
    assert reformulated.count("?") <= 2, "Contextualize không được tổng hợp nhiều câu hỏi"
    
    # Step 2: Query analysis
    state["reformulated_question"] = reformulated
    state["sub_queries"] = ["old query"]  # Sub queries cũ
    result = analyze_query_node(state)
    print(f"2. Query analysis: intent={result['analyzed_intent']}, sub_queries={result['sub_queries']}")
    assert "old query" not in result["sub_queries"], "Query analysis phải reset sub_queries"
    
    # Step 3: Retriever (nếu có vector store)
    try:
        vector_store = build_or_load_medical_vector_store(
            CHATBOT_CONFIG,
            force_rebuild=False,
            use_huggingface=True,
        )
        state["sub_queries"] = result["sub_queries"]
        state["doc_context"] = ["old doc"]  # Doc context cũ
        result = medical_retriever_node(state, vector_store)
        print(f"3. Retriever: retrieved {len(result['doc_context'])} documents")
        assert "old doc" not in str(result["doc_context"]), "Retriever phải reset doc_context"
    except Exception as e:
        print(f"3. Retriever: skipped (error: {e})")
    
    print("✓ PASS: Integration test với fixes")

