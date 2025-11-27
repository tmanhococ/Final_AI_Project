"""Factory utilities to create Google GenAI LLM and embeddings for production.

Tách riêng factory giúp:
- Dễ cấu hình theo `CHATBOT_CONFIG`.
- Dễ mocking trong unit test (không gọi API thật).
"""

from __future__ import annotations

from typing import Optional

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from src.chatbot.config import CHATBOT_CONFIG, ChatbotConfig


def configure_google_client(config: Optional[ChatbotConfig] = None) -> ChatbotConfig:
    """Ensure Google GenAI client is configured with API key from config."""
    cfg = config or CHATBOT_CONFIG
    if not cfg.google_api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not configured. "
            "Hãy thêm vào file .env trong thư mục chatbot hoặc project root."
        )
    genai.configure(api_key=cfg.google_api_key)
    return cfg


def create_production_llm(config: Optional[ChatbotConfig] = None) -> ChatGoogleGenerativeAI:
    """Create ChatGoogleGenerativeAI instance using project config."""
    cfg = configure_google_client(config)
    return ChatGoogleGenerativeAI(
        model=cfg.llm_model_name,
        temperature=0.2,
    )


def create_production_embeddings(
    config: Optional[ChatbotConfig] = None,
) -> GoogleGenerativeAIEmbeddings:
    """Create GoogleGenerativeAIEmbeddings instance using project config."""
    cfg = configure_google_client(config)
    return GoogleGenerativeAIEmbeddings(model=cfg.embedding_model_name)



