"""
Evaluation Metrics for RAG Chatbot.

Provides two groups of metrics:
1. Traditional Metrics (require Ground Truth): BLEU, ROUGE, BERTScore
2. RAG-Specific Metrics (LLM-as-Judge): Faithfulness, Answer Relevancy, 
   Context Precision, Context Recall

Author: AI Evaluation Framework
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import rate limiter
from .rate_limiter import wait_for_rate_limit


# ============================================================================
# Base Metric Class
# ============================================================================

@dataclass
class MetricResult:
    """Result of a metric evaluation."""
    name: str
    score: float
    details: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        return f"{self.name}: {self.score:.4f}"


class BaseMetric(ABC):
    """Abstract base class for all metrics."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the metric."""
        pass
    
    @property
    @abstractmethod
    def requires_ground_truth(self) -> bool:
        """Whether this metric requires ground truth reference."""
        pass
    
    @abstractmethod
    def compute(self, **kwargs) -> MetricResult:
        """Compute the metric score."""
        pass


# ============================================================================
# Traditional Metrics (Require Ground Truth)
# ============================================================================

class BLEUScore(BaseMetric):
    """
    BLEU Score - Bilingual Evaluation Understudy.
    
    Measures n-gram precision between generated answer and ground truth.
    Higher score = more lexical overlap with reference.
    
    Limitations for RAG: Punishes valid paraphrasing.
    """
    
    @property
    def name(self) -> str:
        return "BLEU"
    
    @property
    def requires_ground_truth(self) -> bool:
        return True
    
    def __init__(self, n_gram: int = 4):
        """
        Initialize BLEU scorer.
        
        Args:
            n_gram: Maximum n-gram to consider (default: 4 for BLEU-4)
        """
        self.n_gram = n_gram
        self._nltk_initialized = False
    
    def _ensure_nltk(self):
        """Ensure NLTK data is downloaded."""
        if not self._nltk_initialized:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            try:
                nltk.data.find('tokenizers/punkt_tab')
            except LookupError:
                nltk.download('punkt_tab', quiet=True)
            self._nltk_initialized = True
    
    def compute(
        self,
        answer: str,
        ground_truth: str,
        **kwargs
    ) -> MetricResult:
        """
        Compute BLEU score between answer and ground truth.
        
        Args:
            answer: Generated answer from chatbot
            ground_truth: Reference answer
            
        Returns:
            MetricResult with BLEU score (0-1)
        """
        self._ensure_nltk()
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.tokenize import word_tokenize
        
        # Tokenize
        try:
            reference_tokens = word_tokenize(ground_truth.lower())
            candidate_tokens = word_tokenize(answer.lower())
        except Exception:
            # Fallback to simple split if tokenization fails
            reference_tokens = ground_truth.lower().split()
            candidate_tokens = answer.lower().split()
        
        # Handle empty cases
        if not reference_tokens or not candidate_tokens:
            return MetricResult(name=self.name, score=0.0)
        
        # Compute BLEU with smoothing (prevents 0 scores for short sentences)
        smoothie = SmoothingFunction().method1
        
        # Weights for different n-grams
        weights = tuple([1.0 / self.n_gram] * self.n_gram)
        
        try:
            score = sentence_bleu(
                [reference_tokens],
                candidate_tokens,
                weights=weights,
                smoothing_function=smoothie
            )
        except Exception:
            score = 0.0
        
        return MetricResult(
            name=self.name,
            score=float(score),
            details={
                "n_gram": self.n_gram,
                "reference_length": len(reference_tokens),
                "candidate_length": len(candidate_tokens)
            }
        )


class ROUGEScore(BaseMetric):
    """
    ROUGE Score - Recall-Oriented Understudy for Gisting Evaluation.
    
    Measures n-gram recall (how much of reference appears in answer).
    Returns ROUGE-1, ROUGE-2, and ROUGE-L scores.
    
    Limitations for RAG: Doesn't capture semantic meaning.
    """
    
    @property
    def name(self) -> str:
        return "ROUGE"
    
    @property
    def requires_ground_truth(self) -> bool:
        return True
    
    def __init__(self):
        """Initialize ROUGE scorer."""
        self._scorer = None
    
    def _ensure_scorer(self):
        """Lazy initialization of ROUGE scorer."""
        if self._scorer is None:
            from rouge_score import rouge_scorer
            self._scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )
    
    def compute(
        self,
        answer: str,
        ground_truth: str,
        **kwargs
    ) -> MetricResult:
        """
        Compute ROUGE scores between answer and ground truth.
        
        Args:
            answer: Generated answer from chatbot
            ground_truth: Reference answer
            
        Returns:
            MetricResult with average ROUGE score and individual scores in details
        """
        self._ensure_scorer()
        
        # Handle empty cases
        if not answer.strip() or not ground_truth.strip():
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
            )
        
        scores = self._scorer.score(ground_truth, answer)
        
        # Extract F1 scores
        rouge1_f1 = scores['rouge1'].fmeasure
        rouge2_f1 = scores['rouge2'].fmeasure
        rougeL_f1 = scores['rougeL'].fmeasure
        
        # Average score
        avg_score = (rouge1_f1 + rouge2_f1 + rougeL_f1) / 3
        
        return MetricResult(
            name=self.name,
            score=float(avg_score),
            details={
                "rouge1_f1": float(rouge1_f1),
                "rouge2_f1": float(rouge2_f1),
                "rougeL_f1": float(rougeL_f1),
                "rouge1_precision": float(scores['rouge1'].precision),
                "rouge1_recall": float(scores['rouge1'].recall),
            }
        )


class BERTScoreMetric(BaseMetric):
    """
    BERTScore - Semantic similarity using BERT embeddings.
    
    Overcomes lexical matching limitations by comparing embeddings.
    Understands that "cat" ≈ "feline" semantically.
    
    Advantage over BLEU/ROUGE: Captures semantic similarity.
    Still requires ground truth reference.
    """
    
    @property
    def name(self) -> str:
        return "BERTScore"
    
    @property
    def requires_ground_truth(self) -> bool:
        return True
    
    def __init__(self, model_type: str = "bert-base-multilingual-cased"):
        """
        Initialize BERTScore metric.
        
        Args:
            model_type: BERT model to use (default: multilingual for Vietnamese)
        """
        self.model_type = model_type
    
    def compute(
        self,
        answer: str,
        ground_truth: str,
        **kwargs
    ) -> MetricResult:
        """
        Compute BERTScore between answer and ground truth.
        
        Args:
            answer: Generated answer from chatbot
            ground_truth: Reference answer
            
        Returns:
            MetricResult with F1 BERTScore
        """
        # Handle empty cases
        if not answer.strip() or not ground_truth.strip():
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"precision": 0.0, "recall": 0.0, "f1": 0.0}
            )
        
        try:
            from bert_score import score as bert_score
            
            P, R, F1 = bert_score(
                [answer],
                [ground_truth],
                model_type=self.model_type,
                verbose=False
            )
            
            return MetricResult(
                name=self.name,
                score=float(F1.mean().item()),
                details={
                    "precision": float(P.mean().item()),
                    "recall": float(R.mean().item()),
                    "f1": float(F1.mean().item()),
                    "model": self.model_type
                }
            )
        except Exception as e:
            # Fallback: simple embedding similarity
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"error": str(e)}
            )


# ============================================================================
# RAG-Specific Metrics (LLM-as-Judge - No Ground Truth Required)
# ============================================================================

class FaithfulnessMetric(BaseMetric):
    """
    Faithfulness - Measures factual consistency with retrieved context.
    
    Uses LLM-as-Judge approach:
    1. Extract claims/statements from the answer
    2. Check if each claim can be inferred from context
    3. Score = |supported claims| / |total claims|
    
    Key Question: "Is the chatbot making things up?"
    """
    
    @property
    def name(self) -> str:
        return "Faithfulness"
    
    @property
    def requires_ground_truth(self) -> bool:
        return False  # Only requires context
    
    def __init__(self, llm=None):
        """
        Initialize Faithfulness metric.
        
        Args:
            llm: LangChain LLM instance (if None, will use Gemini from config)
        """
        self._llm = llm
    
    def _get_llm(self):
        """Get or create LLM instance."""
        if self._llm is None:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from src.chatbot.config import CHATBOT_CONFIG
            
            self._llm = ChatGoogleGenerativeAI(
                model=CHATBOT_CONFIG.llm_model_name,
                google_api_key=CHATBOT_CONFIG.google_api_key,
                temperature=0.0  # Deterministic for evaluation
            )
        return self._llm
    
    def _extract_claims(self, answer: str) -> List[str]:
        """
        Extract factual claims from the answer using LLM.
        
        Args:
            answer: The chatbot's response
            
        Returns:
            List of claims/statements
        """
        llm = self._get_llm()
        
        prompt = f"""Hãy trích xuất các câu nhận định/khẳng định thực tế (factual claims) từ đoạn văn sau.
Mỗi claim nên là một câu đơn, độc lập, có thể kiểm chứng được.
Trả về dạng danh sách, mỗi claim trên một dòng, bắt đầu bằng dấu "-".

Đoạn văn:
\"\"\"
{answer}
\"\"\"

Danh sách claims:"""
        
        try:
            wait_for_rate_limit()
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse claims from response
            claims = []
            for line in content.strip().split('\n'):
                line = line.strip()
                if line.startswith('-'):
                    claim = line[1:].strip()
                    if claim:
                        claims.append(claim)
            
            return claims if claims else [answer.strip()]  # Fallback: treat whole answer as one claim
            
        except Exception:
            # Fallback: simple sentence splitting
            sentences = re.split(r'[.!?]+', answer)
            return [s.strip() for s in sentences if s.strip()]
    
    def _verify_claim(self, claim: str, context: str) -> bool:
        """
        Check if a claim is supported by the context.
        
        Args:
            claim: A single factual claim
            context: The retrieved context
            
        Returns:
            True if claim is supported, False otherwise
        """
        llm = self._get_llm()
        
        prompt = f"""Bạn là người đánh giá trung thực. Hãy kiểm tra xem câu nhận định sau có thể được suy ra từ ngữ cảnh hay không.

Ngữ cảnh:
\"\"\"
{context}
\"\"\"

Câu nhận định:
\"\"\"
{claim}
\"\"\"

Trả lời CHỈ MỘT từ: "YES" nếu câu nhận định được hỗ trợ bởi ngữ cảnh, "NO" nếu không.
Trả lời:"""
        
        try:
            wait_for_rate_limit()
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            return "YES" in content.upper()
        except Exception:
            return False
    
    def compute(
        self,
        answer: str,
        contexts: List[str],
        **kwargs
    ) -> MetricResult:
        """
        Compute Faithfulness score.
        
        Args:
            answer: Generated answer from chatbot
            contexts: List of retrieved context chunks
            
        Returns:
            MetricResult with Faithfulness score (0-1)
        """
        if not answer.strip():
            return MetricResult(name=self.name, score=0.0)
        
        if not contexts:
            return MetricResult(
                name=self.name, 
                score=0.0,
                details={"error": "No context provided"}
            )
        
        # Combine all contexts
        combined_context = "\n\n".join(contexts)
        
        # Extract claims
        claims = self._extract_claims(answer)
        
        if not claims:
            return MetricResult(name=self.name, score=1.0, details={"claims": 0})
        
        # Verify each claim
        supported_count = 0
        claim_results = []
        
        for claim in claims:
            is_supported = self._verify_claim(claim, combined_context)
            if is_supported:
                supported_count += 1
            claim_results.append({"claim": claim, "supported": is_supported})
        
        score = supported_count / len(claims)
        
        return MetricResult(
            name=self.name,
            score=float(score),
            details={
                "total_claims": len(claims),
                "supported_claims": supported_count,
                "unsupported_claims": len(claims) - supported_count,
                "claim_details": claim_results
            }
        )


class AnswerRelevancyMetric(BaseMetric):
    """
    Answer Relevancy - Measures if answer addresses the question.
    
    Uses reverse engineering approach:
    1. Generate N hypothetical questions that the answer would address
    2. Compare similarity between original question and generated questions
    3. Score = average cosine similarity
    
    Key Question: "Is the answer on-topic?"
    """
    
    @property
    def name(self) -> str:
        return "Answer Relevancy"
    
    @property
    def requires_ground_truth(self) -> bool:
        return False
    
    def __init__(self, llm=None, num_questions: int = 3):
        """
        Initialize Answer Relevancy metric.
        
        Args:
            llm: LangChain LLM instance
            num_questions: Number of hypothetical questions to generate
        """
        self._llm = llm
        self._embeddings = None
        self.num_questions = num_questions
    
    def _get_llm(self):
        """Get or create LLM instance."""
        if self._llm is None:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from src.chatbot.config import CHATBOT_CONFIG
            
            self._llm = ChatGoogleGenerativeAI(
                model=CHATBOT_CONFIG.llm_model_name,
                google_api_key=CHATBOT_CONFIG.google_api_key,
                temperature=0.3
            )
        return self._llm
    
    def _get_embeddings(self):
        """Get or create embeddings model."""
        if self._embeddings is None:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            from src.chatbot.config import CHATBOT_CONFIG
            
            self._embeddings = GoogleGenerativeAIEmbeddings(
                model=CHATBOT_CONFIG.embedding_model_name,
                google_api_key=CHATBOT_CONFIG.google_api_key
            )
        return self._embeddings
    
    def _generate_questions(self, answer: str) -> List[str]:
        """
        Generate hypothetical questions that the answer would address.
        
        Args:
            answer: The chatbot's response
            
        Returns:
            List of generated questions
        """
        llm = self._get_llm()
        
        prompt = f"""Dựa vào câu trả lời sau, hãy sinh ra {self.num_questions} câu hỏi mà câu trả lời này có thể giải đáp.
Mỗi câu hỏi trên một dòng, bắt đầu bằng số thứ tự.

Câu trả lời:
\"\"\"
{answer}
\"\"\"

Các câu hỏi:"""
        
        try:
            wait_for_rate_limit()
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse questions
            questions = []
            for line in content.strip().split('\n'):
                line = line.strip()
                # Remove numbering
                cleaned = re.sub(r'^[\d]+[.):\s]+', '', line).strip()
                if cleaned and '?' in cleaned:
                    questions.append(cleaned)
            
            return questions[:self.num_questions]
            
        except Exception:
            return []
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a = np.array(vec1)
        b = np.array(vec2)
        
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def compute(
        self,
        answer: str,
        question: str,
        **kwargs
    ) -> MetricResult:
        """
        Compute Answer Relevancy score.
        
        Args:
            answer: Generated answer from chatbot
            question: Original user question
            
        Returns:
            MetricResult with Relevancy score (0-1)
        """
        if not answer.strip() or not question.strip():
            return MetricResult(name=self.name, score=0.0)
        
        # Generate hypothetical questions
        generated_questions = self._generate_questions(answer)
        
        if not generated_questions:
            return MetricResult(
                name=self.name,
                score=0.5,  # Neutral score if generation fails
                details={"error": "Failed to generate questions"}
            )
        
        try:
            embeddings = self._get_embeddings()
            
            # Embed original question
            original_embedding = embeddings.embed_query(question)
            
            # Embed generated questions and compute similarities
            similarities = []
            for gen_q in generated_questions:
                gen_embedding = embeddings.embed_query(gen_q)
                sim = self._cosine_similarity(original_embedding, gen_embedding)
                similarities.append(sim)
            
            avg_score = sum(similarities) / len(similarities)
            
            return MetricResult(
                name=self.name,
                score=float(avg_score),
                details={
                    "generated_questions": generated_questions,
                    "similarities": similarities,
                    "original_question": question
                }
            )
            
        except Exception as e:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"error": str(e)}
            )


class ContextPrecisionMetric(BaseMetric):
    """
    Context Precision - Measures signal-to-noise ratio in retrieved chunks.
    
    Evaluates if relevant chunks are ranked higher than irrelevant ones.
    Uses Average Precision formula.
    
    Key Question: "Is the retrieval result noisy?"
    """
    
    @property
    def name(self) -> str:
        return "Context Precision"
    
    @property
    def requires_ground_truth(self) -> bool:
        return True  # Needs ground truth to determine relevance
    
    def __init__(self, llm=None):
        """
        Initialize Context Precision metric.
        
        Args:
            llm: LangChain LLM instance for relevance judgment
        """
        self._llm = llm
    
    def _get_llm(self):
        """Get or create LLM instance."""
        if self._llm is None:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from src.chatbot.config import CHATBOT_CONFIG
            
            self._llm = ChatGoogleGenerativeAI(
                model=CHATBOT_CONFIG.llm_model_name,
                google_api_key=CHATBOT_CONFIG.google_api_key,
                temperature=0.0
            )
        return self._llm
    
    def _is_context_relevant(self, context: str, ground_truth: str) -> bool:
        """
        Check if a context chunk is relevant based on ground truth.
        
        Args:
            context: A single context chunk
            ground_truth: Reference answer
            
        Returns:
            True if context is relevant
        """
        llm = self._get_llm()
        
        prompt = f"""Hãy đánh giá xem đoạn ngữ cảnh sau có chứa thông tin hữu ích để trả lời được câu trả lời mẫu hay không.

Ngữ cảnh:
\"\"\"
{context}
\"\"\"

Câu trả lời mẫu (Ground Truth):
\"\"\"
{ground_truth}
\"\"\"

Trả lời CHỈ MỘT từ: "YES" nếu ngữ cảnh liên quan và hữu ích, "NO" nếu không.
Trả lời:"""
        
        try:
            wait_for_rate_limit()
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            return "YES" in content.upper()
        except Exception:
            return False
    
    def compute(
        self,
        contexts: List[str],
        ground_truth: str,
        **kwargs
    ) -> MetricResult:
        """
        Compute Context Precision score using Average Precision.
        
        Args:
            contexts: List of retrieved context chunks (ordered by rank)
            ground_truth: Reference answer
            
        Returns:
            MetricResult with Precision score (0-1)
        """
        if not contexts:
            return MetricResult(name=self.name, score=0.0)
        
        if not ground_truth.strip():
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"error": "No ground truth provided"}
            )
        
        # Evaluate relevance for each context
        relevance = []
        for ctx in contexts:
            is_relevant = self._is_context_relevant(ctx, ground_truth)
            relevance.append(1 if is_relevant else 0)
        
        # Compute Average Precision
        # AP = sum(Precision@k * v_k) / total_relevant
        total_relevant = sum(relevance)
        
        if total_relevant == 0:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"relevance": relevance, "total_relevant": 0}
            )
        
        precision_sum = 0.0
        relevant_count = 0
        
        for k, is_rel in enumerate(relevance, 1):
            if is_rel:
                relevant_count += 1
                precision_at_k = relevant_count / k
                precision_sum += precision_at_k
        
        average_precision = precision_sum / total_relevant
        
        return MetricResult(
            name=self.name,
            score=float(average_precision),
            details={
                "relevance_by_rank": relevance,
                "total_contexts": len(contexts),
                "relevant_contexts": total_relevant
            }
        )


class ContextRecallMetric(BaseMetric):
    """
    Context Recall - Measures if all necessary information was retrieved.
    
    Analyzes ground truth into key points and checks if each appears in context.
    
    Key Question: "Did we miss any important information?"
    """
    
    @property
    def name(self) -> str:
        return "Context Recall"
    
    @property
    def requires_ground_truth(self) -> bool:
        return True
    
    def __init__(self, llm=None):
        """
        Initialize Context Recall metric.
        
        Args:
            llm: LangChain LLM instance
        """
        self._llm = llm
    
    def _get_llm(self):
        """Get or create LLM instance."""
        if self._llm is None:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from src.chatbot.config import CHATBOT_CONFIG
            
            self._llm = ChatGoogleGenerativeAI(
                model=CHATBOT_CONFIG.llm_model_name,
                google_api_key=CHATBOT_CONFIG.google_api_key,
                temperature=0.0
            )
        return self._llm
    
    def _extract_key_points(self, ground_truth: str) -> List[str]:
        """
        Extract key points from ground truth.
        
        Args:
            ground_truth: Reference answer
            
        Returns:
            List of key points
        """
        llm = self._get_llm()
        
        prompt = f"""Hãy trích xuất các ý chính (key points) từ câu trả lời mẫu sau.
Mỗi ý chính trên một dòng, bắt đầu bằng dấu "-".

Câu trả lời mẫu:
\"\"\"
{ground_truth}
\"\"\"

Các ý chính:"""
        
        try:
            wait_for_rate_limit()
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            key_points = []
            for line in content.strip().split('\n'):
                line = line.strip()
                if line.startswith('-'):
                    point = line[1:].strip()
                    if point:
                        key_points.append(point)
            
            return key_points if key_points else [ground_truth.strip()]
            
        except Exception:
            # Fallback: simple sentence splitting
            sentences = re.split(r'[.!?]+', ground_truth)
            return [s.strip() for s in sentences if s.strip()]
    
    def _is_point_in_context(self, key_point: str, context: str) -> bool:
        """
        Check if a key point can be found in the context.
        
        Args:
            key_point: A single key point from ground truth
            context: Combined context from retriever
            
        Returns:
            True if key point is found in context
        """
        llm = self._get_llm()
        
        prompt = f"""Hãy kiểm tra xem ý chính sau có xuất hiện hoặc có thể suy ra từ ngữ cảnh hay không.

Ngữ cảnh:
\"\"\"
{context}
\"\"\"

Ý chính cần kiểm tra:
\"\"\"
{key_point}
\"\"\"

Trả lời CHỈ MỘT từ: "YES" nếu ý chính được tìm thấy trong ngữ cảnh, "NO" nếu không.
Trả lời:"""
        
        try:
            wait_for_rate_limit()
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            return "YES" in content.upper()
        except Exception:
            return False
    
    def compute(
        self,
        contexts: List[str],
        ground_truth: str,
        **kwargs
    ) -> MetricResult:
        """
        Compute Context Recall score.
        
        Args:
            contexts: List of retrieved context chunks
            ground_truth: Reference answer
            
        Returns:
            MetricResult with Recall score (0-1)
        """
        if not ground_truth.strip():
            return MetricResult(name=self.name, score=0.0)
        
        if not contexts:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"error": "No context provided"}
            )
        
        # Combine contexts
        combined_context = "\n\n".join(contexts)
        
        # Extract key points from ground truth
        key_points = self._extract_key_points(ground_truth)
        
        if not key_points:
            return MetricResult(name=self.name, score=1.0, details={"key_points": 0})
        
        # Check each key point
        found_count = 0
        point_results = []
        
        for point in key_points:
            is_found = self._is_point_in_context(point, combined_context)
            if is_found:
                found_count += 1
            point_results.append({"point": point, "found": is_found})
        
        score = found_count / len(key_points)
        
        return MetricResult(
            name=self.name,
            score=float(score),
            details={
                "total_key_points": len(key_points),
                "found_key_points": found_count,
                "missed_key_points": len(key_points) - found_count,
                "point_details": point_results
            }
        )


# ============================================================================
# Utility Functions
# ============================================================================

def get_all_metrics(llm=None) -> Dict[str, BaseMetric]:
    """
    Get all available metrics.
    
    Args:
        llm: Optional LLM instance to share across metrics
        
    Returns:
        Dictionary of metric name -> metric instance
    """
    return {
        # Traditional Metrics
        "BLEU": BLEUScore(),
        "ROUGE": ROUGEScore(),
        "BERTScore": BERTScoreMetric(),
        # RAG Metrics
        "Faithfulness": FaithfulnessMetric(llm=llm),
        "Answer Relevancy": AnswerRelevancyMetric(llm=llm),
        "Context Precision": ContextPrecisionMetric(llm=llm),
        "Context Recall": ContextRecallMetric(llm=llm),
    }


def get_traditional_metrics() -> Dict[str, BaseMetric]:
    """Get only traditional metrics (BLEU, ROUGE, BERTScore)."""
    return {
        "BLEU": BLEUScore(),
        "ROUGE": ROUGEScore(),
        "BERTScore": BERTScoreMetric(),
    }


def get_rag_metrics(llm=None) -> Dict[str, BaseMetric]:
    """Get only RAG-specific metrics."""
    return {
        "Faithfulness": FaithfulnessMetric(llm=llm),
        "Answer Relevancy": AnswerRelevancyMetric(llm=llm),
        "Context Precision": ContextPrecisionMetric(llm=llm),
        "Context Recall": ContextRecallMetric(llm=llm),
    }
