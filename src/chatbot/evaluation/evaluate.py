"""
RAG Chatbot Evaluation - Main Entry Point.

Menu-driven interface for running evaluations:
- Generate synthetic test data
- Evaluate individual metrics
- Run batch evaluations
- Generate visualization reports

Usage:
    cd src/chatbot/evaluation
    python evaluate.py

Author: AI Evaluation Framework
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

# Ensure UTF-8 encoding for Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

if sys.stderr.encoding != "utf-8":
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def print_banner():
    """Print the evaluation framework banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║           RAG Chatbot Evaluation Framework                     ║
║               - Final_AI_Project -                             ║
╠═══════════════════════════════════════════════════════════════╣
║  Comprehensive evaluation for RAG-based healthcare chatbot     ║
║  Metrics: BLEU, ROUGE, BERTScore, Faithfulness, Relevancy...   ║
╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_menu():
    """Print the main menu."""
    menu = """
╔═══════════════════════════════════════════════════════════════╗
║                       MAIN MENU                                ║
╠═══════════════════════════════════════════════════════════════╣
║  --- Test Data ---                                             ║
║  1.  Generate Synthetic Test Data                              ║
║  2.  Use Manual Test Set (Pre-defined questions)               ║
║                                                                ║
║  --- Traditional Metrics (require Ground Truth) ---            ║
║  3.  Evaluate BLEU Score                                       ║
║  4.  Evaluate ROUGE Score                                      ║
║  5.  Evaluate BERTScore                                        ║
║                                                                ║
║  --- RAG Metrics (LLM-as-Judge) ---                            ║
║  6.  Evaluate Faithfulness                                     ║
║  7.  Evaluate Answer Relevancy                                 ║
║  8.  Evaluate Context Precision                                ║
║  9.  Evaluate Context Recall                                   ║
║                                                                ║
║  --- Batch Operations ---                                      ║
║  10. Run All Traditional Metrics                               ║
║  11. Run All RAG Metrics                                       ║
║  12. Run ALL Evaluations                                       ║
║  13. Generate Visualization Report                             ║
║                                                                ║
║  0.  Exit                                                      ║
╚═══════════════════════════════════════════════════════════════╝
    """
    print(menu)


def ensure_dependencies():
    """Check and install required dependencies."""
    print("\n⏳ Checking dependencies...")
    
    required = ['nltk', 'rouge_score', 'matplotlib', 'seaborn', 'sklearn']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg.replace('-', '_'))
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"⚠️  Missing packages: {missing}")
        print("   Install with: pip install -r requirements_eval.txt")
        return False
    
    print("✅ All dependencies available")
    return True


def option_1_generate_synthetic():
    """Generate synthetic test data from medical documents."""
    from src.chatbot.evaluation.testset_generator import TestsetGenerator
    
    print("\n" + "="*60)
    print("📝 GENERATE SYNTHETIC TEST DATA")
    print("="*60)
    
    # Get medical docs directory
    docs_dir = PROJECT_ROOT / "src" / "data" / "medical_docs"
    
    if not docs_dir.exists():
        print(f"❌ Directory not found: {docs_dir}")
        return None
    
    print(f"📂 Loading documents from: {docs_dir}")
    
    # Ask for test size
    try:
        test_size = int(input("Enter number of test cases to generate (default: 10): ") or "10")
    except ValueError:
        test_size = 10
    
    generator = TestsetGenerator()
    generator.load_documents_from_directory(docs_dir)
    
    print(f"\n🔄 Generating {test_size} test cases...")
    test_cases = generator.generate(test_size=test_size, verbose=True)
    
    # Save to file
    output_path = Path(__file__).parent / "golden_dataset.json"
    generator.save_testset(test_cases, output_path)
    
    return test_cases


def option_2_manual_testset():
    """Load pre-defined manual test set."""
    from src.chatbot.evaluation.testset_generator import create_manual_testset
    
    print("\n" + "="*60)
    print("📋 USING MANUAL TEST SET")
    print("="*60)
    
    test_cases = create_manual_testset()
    print(f"✅ Loaded {len(test_cases)} pre-defined test cases")
    
    print("\nTest cases by type:")
    types = {}
    for tc in test_cases:
        types[tc.evolution_type] = types.get(tc.evolution_type, 0) + 1
    
    for t, count in types.items():
        print(f"  - {t}: {count}")
    
    return test_cases


def run_single_metric_evaluation(metric_name: str, test_cases):
    """Run evaluation for a single metric."""
    from src.chatbot.evaluation.evaluator import RAGEvaluator
    
    print(f"\n🔄 Running {metric_name} evaluation...")
    
    evaluator = RAGEvaluator(verbose=True)
    batch_result = evaluator.evaluate_batch(test_cases, metrics_to_run=[metric_name])
    
    # Save results
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    safe_name = metric_name.lower().replace(' ', '_')
    batch_result.save_to_csv(output_dir / f"{safe_name}_results.csv")
    
    return batch_result


def option_10_traditional_metrics(test_cases):
    """Run all traditional metrics (BLEU, ROUGE, BERTScore)."""
    from src.chatbot.evaluation.evaluator import RAGEvaluator
    
    print("\n" + "="*60)
    print("📊 RUNNING ALL TRADITIONAL METRICS")
    print("="*60)
    
    metrics = ["BLEU", "ROUGE", "BERTScore"]
    
    evaluator = RAGEvaluator(verbose=True)
    batch_result = evaluator.evaluate_batch(test_cases, metrics_to_run=metrics)
    
    # Save results
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    batch_result.save_to_csv(output_dir / "traditional_metrics_results.csv")
    
    return batch_result


def option_11_rag_metrics(test_cases):
    """Run all RAG-specific metrics."""
    from src.chatbot.evaluation.evaluator import RAGEvaluator
    
    print("\n" + "="*60)
    print("📊 RUNNING ALL RAG METRICS")
    print("="*60)
    
    metrics = ["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall"]
    
    evaluator = RAGEvaluator(verbose=True)
    batch_result = evaluator.evaluate_batch(test_cases, metrics_to_run=metrics)
    
    # Save results
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    batch_result.save_to_csv(output_dir / "rag_metrics_results.csv")
    
    return batch_result


def option_12_run_all(test_cases):
    """Run all evaluations (Traditional + RAG)."""
    from src.chatbot.evaluation.evaluator import RAGEvaluator
    
    print("\n" + "="*60)
    print("📊 RUNNING ALL EVALUATIONS")
    print("="*60)
    
    evaluator = RAGEvaluator(verbose=True)
    batch_result = evaluator.evaluate_batch(test_cases)  # All metrics
    
    # Save results
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    batch_result.save_to_csv(output_dir / "all_metrics_results.csv")
    batch_result.save_to_json(output_dir / "all_metrics_results.json")
    
    return batch_result


def option_13_generate_visualizations(batch_result=None):
    """Generate visualization report."""
    from src.chatbot.evaluation.visualizer import EvaluationVisualizer
    import pandas as pd
    
    print("\n" + "="*60)
    print("📈 GENERATING VISUALIZATION REPORT")
    print("="*60)
    
    output_dir = Path(__file__).parent / "output"
    
    if batch_result is None:
        # Try to load from CSV
        csv_path = output_dir / "all_metrics_results.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            
            # Compute aggregate scores from DataFrame
            metric_cols = [
                col for col in df.columns 
                if df[col].dtype in ['float64', 'float32']
                and col not in ['latency_ms']
            ]
            aggregate_scores = {col: df[col].mean() for col in metric_cols}
        else:
            print("❌ No evaluation results found. Run evaluations first (option 12).")
            return
    else:
        df = batch_result.to_dataframe()
        aggregate_scores = batch_result.aggregate_scores
    
    visualizer = EvaluationVisualizer(output_dir=output_dir)
    charts = visualizer.generate_full_report(df, aggregate_scores)
    
    print(f"\n✅ Generated {len(charts)} visualization charts in: {output_dir}")
    return charts


def main():
    """Main entry point for the evaluation framework."""
    print_banner()
    
    # Check dependencies
    if not ensure_dependencies():
        print("\n⚠️  Some dependencies are missing. Some features may not work.")
    
    test_cases = None
    batch_result = None
    
    while True:
        print_menu()
        
        try:
            choice = input("Enter your choice (0-13): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 👋")
            break
        
        if choice == "0":
            print("\nGoodbye! 👋")
            break
        
        elif choice == "1":
            test_cases = option_1_generate_synthetic()
        
        elif choice == "2":
            test_cases = option_2_manual_testset()
        
        elif choice in ["3", "4", "5", "6", "7", "8", "9"]:
            if test_cases is None:
                print("\n⚠️  No test cases loaded. Please run option 1 or 2 first.")
                continue
            
            metric_map = {
                "3": "BLEU",
                "4": "ROUGE",
                "5": "BERTScore",
                "6": "Faithfulness",
                "7": "Answer Relevancy",
                "8": "Context Precision",
                "9": "Context Recall"
            }
            batch_result = run_single_metric_evaluation(metric_map[choice], test_cases)
        
        elif choice == "10":
            if test_cases is None:
                print("\n⚠️  No test cases loaded. Please run option 1 or 2 first.")
                continue
            batch_result = option_10_traditional_metrics(test_cases)
        
        elif choice == "11":
            if test_cases is None:
                print("\n⚠️  No test cases loaded. Please run option 1 or 2 first.")
                continue
            batch_result = option_11_rag_metrics(test_cases)
        
        elif choice == "12":
            if test_cases is None:
                print("\n⚠️  No test cases loaded. Please run option 1 or 2 first.")
                continue
            batch_result = option_12_run_all(test_cases)
        
        elif choice == "13":
            option_13_generate_visualizations(batch_result)
        
        else:
            print("\n❌ Invalid choice. Please enter a number from 0 to 13.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
