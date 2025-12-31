"""
RAG Evaluator - Core evaluation engine for RAG Chatbot.

Integrates with the existing chatbot to run batch evaluations on test datasets
and compute all metrics (Traditional + RAG-specific).

Author: AI Evaluation Framework
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .metrics import (
    BaseMetric,
    MetricResult,
    BLEUScore,
    ROUGEScore,
    BERTScoreMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
    get_all_metrics,
    get_traditional_metrics,
    get_rag_metrics,
)
from .testset_generator import TestCase


@dataclass
class EvaluationResult:
    """Result of a single test case evaluation."""
    test_case: TestCase
    answer: str
    contexts: List[str]
    metrics: Dict[str, MetricResult] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "question": self.test_case.question,
            "ground_truth": self.test_case.ground_truth,
            "answer": self.answer,
            "contexts": self.contexts,
            "evolution_type": self.test_case.evolution_type,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "metrics": {
                name: {"score": result.score, "details": result.details}
                for name, result in self.metrics.items()
            }
        }


@dataclass
class BatchEvaluationResult:
    """Result of batch evaluation across all test cases."""
    results: List[EvaluationResult]
    aggregate_scores: Dict[str, float] = field(default_factory=dict)
    evaluation_time: str = ""
    total_time_seconds: float = 0.0
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        rows = []
        for result in self.results:
            row = {
                "question": result.test_case.question,
                "ground_truth": result.test_case.ground_truth,
                "answer": result.answer,
                "evolution_type": result.test_case.evolution_type,
                "latency_ms": result.latency_ms,
                "error": result.error,
            }
            # Add metric scores
            for metric_name, metric_result in result.metrics.items():
                row[metric_name] = metric_result.score
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def save_to_csv(self, output_path: str | Path) -> None:
        """Save results to CSV file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df = self.to_dataframe()
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Results saved to {output_path}")
    
    def save_to_json(self, output_path: str | Path) -> None:
        """Save detailed results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "evaluation_time": self.evaluation_time,
            "total_cases": len(self.results),
            "total_time_seconds": self.total_time_seconds,
            "aggregate_scores": self.aggregate_scores,
            "results": [r.to_dict() for r in self.results]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Detailed results saved to {output_path}")


class RAGEvaluator:
    """
    Core evaluation engine for RAG Chatbot.
    
    Integrates with the existing chatbot implementation to:
    1. Run inference on test cases
    2. Capture answers and retrieved contexts
    3. Compute all metrics
    4. Generate reports
    """
    
    def __init__(
        self,
        chatbot_app=None,
        metrics: Optional[Dict[str, BaseMetric]] = None,
        verbose: bool = True
    ):
        """
        Initialize the RAG Evaluator.
        
        Args:
            chatbot_app: The compiled LangGraph chatbot app
                        (if None, will be loaded from app_runtime)
            metrics: Dictionary of metrics to use
                    (if None, will use all available metrics)
            verbose: Print progress messages
        """
        self._chatbot_app = chatbot_app
        self._metrics = metrics
        self.verbose = verbose
        self._thread_id = "evaluation_thread"
    
    def _get_chatbot_app(self):
        """Get or create chatbot app."""
        if self._chatbot_app is None:
            from src.chatbot.app_runtime import create_chatbot_app
            self._chatbot_app = create_chatbot_app()
        return self._chatbot_app
    
    def _get_metrics(self) -> Dict[str, BaseMetric]:
        """Get or create metrics dictionary."""
        if self._metrics is None:
            self._metrics = get_all_metrics()
        return self._metrics
    
    def _run_inference(
        self, 
        question: str,
        thread_id: Optional[str] = None
    ) -> Tuple[str, List[str], float]:
        """
        Run chatbot inference on a single question.
        
        Args:
            question: The question to ask
            thread_id: Thread ID for conversation state
            
        Returns:
            Tuple of (answer, contexts, latency_ms)
        """
        import time
        
        app = self._get_chatbot_app()
        
        # Prepare input state
        from langchain_core.messages import HumanMessage
        
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "original_question": question,
            "reformulated_question": "",
            "generation": "",
            "analyzed_intent": "fall_back",
            "sub_queries": [],
            "context": [],
            "csv_context": [],
            "doc_context": [],
            "retry_count": 0,
            "answer_valid": True,
        }
        
        config = {"configurable": {"thread_id": thread_id or self._thread_id}}
        
        start_time = time.time()
        
        try:
            # Run the graph
            final_state = app.invoke(initial_state, config=config)
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract answer and contexts
            answer = final_state.get("generation", "")
            
            # Combine all context sources
            contexts = []
            if final_state.get("csv_context"):
                contexts.extend(final_state["csv_context"])
            if final_state.get("doc_context"):
                contexts.extend(final_state["doc_context"])
            if final_state.get("context"):
                contexts.extend(final_state["context"])
            
            # Deduplicate
            contexts = list(dict.fromkeys(contexts))
            
            return answer, contexts, latency_ms
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return f"Error: {str(e)}", [], latency_ms
    
    def evaluate_single(
        self,
        test_case: TestCase,
        metrics_to_run: Optional[List[str]] = None
    ) -> EvaluationResult:
        """
        Evaluate a single test case.
        
        Args:
            test_case: The test case to evaluate
            metrics_to_run: List of metric names to run (None = all)
            
        Returns:
            EvaluationResult with computed metrics
        """
        # Run inference
        answer, contexts, latency_ms = self._run_inference(test_case.question)
        
        # Use test case contexts if inference didn't return any
        if not contexts and test_case.contexts:
            contexts = test_case.contexts
        
        result = EvaluationResult(
            test_case=test_case,
            answer=answer,
            contexts=contexts,
            latency_ms=latency_ms
        )
        
        if answer.startswith("Error:"):
            result.error = answer
            return result
        
        # Get metrics to run
        all_metrics = self._get_metrics()
        if metrics_to_run:
            metrics_to_use = {k: v for k, v in all_metrics.items() if k in metrics_to_run}
        else:
            metrics_to_use = all_metrics
        
        # Compute each metric
        for name, metric in metrics_to_use.items():
            try:
                if metric.requires_ground_truth:
                    if not test_case.ground_truth:
                        continue  # Skip if no ground truth available
                    
                    if name in ["BLEU", "ROUGE", "BERTScore"]:
                        metric_result = metric.compute(
                            answer=answer,
                            ground_truth=test_case.ground_truth
                        )
                    elif name == "Context Precision":
                        metric_result = metric.compute(
                            contexts=contexts,
                            ground_truth=test_case.ground_truth
                        )
                    elif name == "Context Recall":
                        metric_result = metric.compute(
                            contexts=contexts,
                            ground_truth=test_case.ground_truth
                        )
                    else:
                        continue
                else:
                    if name == "Faithfulness":
                        metric_result = metric.compute(
                            answer=answer,
                            contexts=contexts
                        )
                    elif name == "Answer Relevancy":
                        metric_result = metric.compute(
                            answer=answer,
                            question=test_case.question
                        )
                    else:
                        continue
                
                result.metrics[name] = metric_result
                
            except Exception as e:
                result.metrics[name] = MetricResult(
                    name=name,
                    score=0.0,
                    details={"error": str(e)}
                )
        
        return result
    
    def evaluate_batch(
        self,
        test_cases: List[TestCase],
        metrics_to_run: Optional[List[str]] = None
    ) -> BatchEvaluationResult:
        """
        Evaluate a batch of test cases.
        
        Args:
            test_cases: List of test cases to evaluate
            metrics_to_run: List of metric names to run (None = all)
            
        Returns:
            BatchEvaluationResult with all results and aggregates
        """
        import time
        
        start_time = time.time()
        results: List[EvaluationResult] = []
        
        for i, test_case in enumerate(test_cases):
            if self.verbose:
                print(f"\n[{i+1}/{len(test_cases)}] Evaluating: {test_case.question[:50]}...")
            
            result = self.evaluate_single(test_case, metrics_to_run)
            results.append(result)
            
            if self.verbose:
                if result.error:
                    print(f"  ❌ Error: {result.error[:50]}...")
                else:
                    scores = [f"{k}: {v.score:.2f}" for k, v in result.metrics.items()]
                    print(f"  ✓ Latency: {result.latency_ms:.0f}ms | Scores: {', '.join(scores)}")
        
        total_time = time.time() - start_time
        
        # Compute aggregate scores (average of each metric)
        aggregate_scores: Dict[str, float] = {}
        all_metric_names = set()
        for r in results:
            all_metric_names.update(r.metrics.keys())
        
        for metric_name in all_metric_names:
            scores = [
                r.metrics[metric_name].score 
                for r in results 
                if metric_name in r.metrics
            ]
            if scores:
                aggregate_scores[metric_name] = sum(scores) / len(scores)
        
        batch_result = BatchEvaluationResult(
            results=results,
            aggregate_scores=aggregate_scores,
            evaluation_time=datetime.now().isoformat(),
            total_time_seconds=total_time
        )
        
        if self.verbose:
            self._print_summary(batch_result)
        
        return batch_result
    
    def _print_summary(self, batch_result: BatchEvaluationResult) -> None:
        """Print evaluation summary to console."""
        print("\n" + "="*60)
        print("              EVALUATION SUMMARY")
        print("="*60)
        print(f"Total test cases: {len(batch_result.results)}")
        print(f"Total time: {batch_result.total_time_seconds:.1f} seconds")
        print(f"Avg latency: {sum(r.latency_ms for r in batch_result.results)/len(batch_result.results):.0f} ms")
        print("-"*60)
        print("\n📊 Aggregate Scores:")
        print("-"*60)
        print(f"{'Metric':<25} {'Score':>10} {'Status':<15}")
        print("-"*60)
        
        for metric_name, score in sorted(batch_result.aggregate_scores.items()):
            status = self._get_status(score)
            print(f"{metric_name:<25} {score:>10.4f} {status:<15}")
        
        print("="*60)
        
        # Print diagnosis based on RAG Triad analysis
        self._print_diagnosis(batch_result.aggregate_scores)
    
    def _get_status(self, score: float) -> str:
        """Get status emoji based on score."""
        if score >= 0.8:
            return "✅ Excellent"
        elif score >= 0.6:
            return "✓ Good"
        elif score >= 0.4:
            return "⚠️ Warning"
        else:
            return "❌ Critical"
    
    def _print_diagnosis(self, scores: Dict[str, float]) -> None:
        """Print diagnosis based on metric scores."""
        faithfulness = scores.get("Faithfulness", None)
        relevancy = scores.get("Answer Relevancy", None)
        
        if faithfulness is not None and relevancy is not None:
            print("\n🔍 Diagnosis (based on RAG Triad analysis):")
            print("-"*60)
            
            if faithfulness >= 0.6 and relevancy >= 0.6:
                print("✅ System is performing well!")
                print("   Continue monitoring for regression.")
            elif faithfulness < 0.6 and relevancy >= 0.6:
                print("⚠️ HALLUCINATION DETECTED!")
                print("   Bot answers confidently but makes up facts.")
                print("   → Reduce LLM temperature")
                print("   → Strengthen system prompt to use context")
            elif faithfulness >= 0.6 and relevancy < 0.6:
                print("⚠️ EVASIVENESS DETECTED!")
                print("   Bot answers truthfully but doesn't address the question.")
                print("   → Improve prompt to be more direct")
                print("   → Check if retriever is getting correct chunks")
            else:
                print("❌ SYSTEM FAILURE!")
                print("   Both faithfulness and relevancy are low.")
                print("   → Check entire pipeline: embedding, retriever, generator")
            
            print("="*60)


def evaluate_single_metric(
    metric_name: str,
    test_cases: List[TestCase],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Convenience function to evaluate a single metric.
    
    Args:
        metric_name: Name of the metric to run
        test_cases: List of test cases
        verbose: Print progress
        
    Returns:
        DataFrame with results
    """
    evaluator = RAGEvaluator(verbose=verbose)
    batch_result = evaluator.evaluate_batch(test_cases, metrics_to_run=[metric_name])
    return batch_result.to_dataframe()
