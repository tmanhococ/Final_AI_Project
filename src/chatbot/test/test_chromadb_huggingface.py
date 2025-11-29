"""Test ChromaDB với HuggingFace embeddings để đảm bảo tương thích."""

from __future__ import annotations

import sys
from pathlib import Path

# Đảm bảo project root trong sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chatbot.tools.vector_store import (
    create_huggingface_embeddings,
    build_or_load_medical_vector_store,
    load_medical_documents,
    build_chroma_from_documents,
    get_default_paths,
)
from src.chatbot.config import CHATBOT_CONFIG


def test_huggingface_with_chromadb():
    """Test: HuggingFace embeddings có thể query ChromaDB."""
    print("="*80)
    print("TEST: HuggingFace embeddings với ChromaDB")
    print("="*80)
    
    # 1. Tạo HuggingFace embeddings
    print("\n1. Tạo HuggingFace embeddings...")
    embeddings = create_huggingface_embeddings()
    print("   ✓ HuggingFace embeddings created")
    
    # 2. Test embed query
    print("\n2. Test embed query...")
    query = "Hội chứng mỏi mắt CVS là gì?"
    query_embedding = embeddings.embed_query(query)
    print(f"   ✓ Query embedded, dimension: {len(query_embedding)}")
    
    # 3. Build hoặc load vector store
    print("\n3. Build/load vector store với HuggingFace embeddings...")
    try:
        vector_store = build_or_load_medical_vector_store(
            CHATBOT_CONFIG,
            force_rebuild=True,  # Force rebuild để test từ đầu
            use_huggingface=True,
        )
        print("   ✓ Vector store created/loaded")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        raise
    
    # 4. Test query
    print("\n4. Test query vector store...")
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    results = retriever.invoke(query)
    print(f"   ✓ Retrieved {len(results)} documents")
    
    # 5. In kết quả
    print("\n5. Kết quả:")
    for i, doc in enumerate(results, 1):
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"   Doc {i}: {preview}...")
    
    # 6. Verify
    assert len(results) > 0, "Phải retrieve được ít nhất 1 document"
    assert len(results[0].page_content) > 0, "Document phải có nội dung"
    
    print("\n" + "="*80)
    print("✓ PASS: HuggingFace embeddings có thể query ChromaDB")
    print("="*80)


if __name__ == "__main__":
    test_huggingface_with_chromadb()

