"""
RAG Chatbot Evaluation Framework.

Provides comprehensive evaluation metrics for RAG-based healthcare chatbot:
- Traditional Metrics: BLEU, ROUGE, BERTScore
- RAG Triad Metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall
"""

from .metrics import (
    BLEUScore,
    ROUGEScore,
    BERTScoreMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
)
from .evaluator import RAGEvaluator
from .visualizer import EvaluationVisualizer
from .testset_generator import TestsetGenerator

__all__ = [
    # Traditional Metrics
    "BLEUScore",
    "ROUGEScore", 
    "BERTScoreMetric",
    # RAG Triad Metrics
    "FaithfulnessMetric",
    "AnswerRelevancyMetric",
    "ContextPrecisionMetric",
    "ContextRecallMetric",
    # Core Classes
    "RAGEvaluator",
    "EvaluationVisualizer",
    "TestsetGenerator",
]
