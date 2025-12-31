# RAG Chatbot Evaluation Framework

Hệ thống đánh giá toàn diện cho Chatbot RAG sử dụng phương pháp **LLM-as-a-Judge** và các metrics truyền thống.

## 📋 Mục lục

- [Cài đặt](#-cài-đặt)
- [Cách sử dụng](#-cách-sử-dụng)
- [Các Metrics](#-các-metrics)
- [Sinh dữ liệu test](#-sinh-dữ-liệu-test)
- [Trực quan hóa](#-trực-quan-hóa)
- [Giải thích kết quả](#-giải-thích-kết-quả)

---

## 🔧 Cài đặt

```bash
# Cài đặt dependencies
cd src/chatbot/evaluation
pip install -r requirements_eval.txt
```

**Dependencies:**
- `nltk` - BLEU score
- `rouge-score` - ROUGE metrics
- `bert-score` - BERTScore (optional, cần PyTorch)
- `matplotlib`, `seaborn` - Visualization
- `scikit-learn` - Cosine similarity

---

## 🚀 Cách sử dụng

### Chạy giao diện menu

```bash
cd d:\AI_Final\Final_AI_Project\src\chatbot\evaluation
python evaluate.py
```

### Menu Options

| Option | Chức năng | Mô tả |
|--------|-----------|-------|
| 1 | Generate Synthetic Data | Tạo dữ liệu test từ medical_docs |
| 2 | Manual Test Set | Dùng bộ test 15 câu hỏi có sẵn |
| 3-5 | Traditional Metrics | BLEU, ROUGE, BERTScore |
| 6-9 | RAG Metrics | Faithfulness, Relevancy, Precision, Recall |
| 10 | All Traditional | Chạy tất cả metrics truyền thống |
| 11 | All RAG | Chạy tất cả RAG metrics |
| 12 | Run All | Chạy toàn bộ evaluation |
| 13 | Visualization | Tạo biểu đồ báo cáo |

### Sử dụng từng module riêng lẻ

```python
# Import metrics
from src.chatbot.evaluation.metrics import (
    BLEUScore, ROUGEScore, BERTScoreMetric,
    FaithfulnessMetric, AnswerRelevancyMetric
)

# Tính BLEU
bleu = BLEUScore()
result = bleu.compute(
    answer="Chatbot trả lời đây",
    ground_truth="Đây là câu trả lời mẫu"
)
print(f"BLEU: {result.score:.4f}")

# Tính Faithfulness (không cần ground truth)
faithfulness = FaithfulnessMetric()
result = faithfulness.compute(
    answer="CVS là hội chứng mỏi mắt do dùng máy tính",
    contexts=["CVS (Computer Vision Syndrome) gây mỏi mắt..."]
)
print(f"Faithfulness: {result.score:.4f}")
```

---

## 📊 Các Metrics

### Nhóm 1: Traditional Metrics (Cần Ground Truth)

| Metric | Công thức | Ý nghĩa |
|--------|-----------|---------|
| **BLEU** | n-gram precision | Đo trùng khớp từ vựng với câu mẫu |
| **ROUGE** | n-gram recall | Đo độ phủ của câu mẫu trong câu trả lời |
| **BERTScore** | Cosine similarity embeddings | Đo tương đồng ngữ nghĩa |

### Nhóm 2: RAG Metrics (LLM-as-Judge)

| Metric | Câu hỏi cốt lõi | Input cần |
|--------|-----------------|-----------|
| **Faithfulness** | Bot có bịa đặt không? | answer, contexts |
| **Answer Relevancy** | Câu trả lời có đúng trọng tâm? | answer, question |
| **Context Precision** | Kết quả tìm kiếm có nhiễu? | contexts, ground_truth |
| **Context Recall** | Có bỏ sót thông tin? | contexts, ground_truth |

---

## 📝 Sinh dữ liệu test

### 1. Sử dụng bộ test thủ công (nhanh)

```python
from src.chatbot.evaluation.testset_generator import create_manual_testset

test_cases = create_manual_testset()
print(f"Loaded {len(test_cases)} test cases")
```

**Bộ test thủ công gồm:**
- 3 câu Social flow (chào hỏi)
- 4 câu CSV flow (phân tích log)
- 5 câu Retriever flow (kiến thức y khoa)
- 3 câu Both flow (kết hợp)

### 2. Sinh dữ liệu tự động từ documents

```python
from src.chatbot.evaluation.testset_generator import TestsetGenerator

generator = TestsetGenerator()
generator.load_documents_from_directory("src/data/medical_docs")

# Sinh 20 câu với phân phối: 50% simple, 30% reasoning, 20% multi-context
test_cases = generator.generate(
    test_size=20,
    distribution={"simple": 0.5, "reasoning": 0.3, "multi_context": 0.2}
)

# Lưu ra file
generator.save_testset(test_cases, "evaluation/golden_dataset.json")
```

---

## 📈 Trực quan hóa

### Tạo báo cáo đầy đủ

```python
from src.chatbot.evaluation.visualizer import EvaluationVisualizer
import pandas as pd

# Load kết quả evaluation
df = pd.read_csv("output/all_metrics_results.csv")
scores = {"Faithfulness": 0.85, "Answer Relevancy": 0.78, ...}

# Tạo tất cả biểu đồ
visualizer = EvaluationVisualizer(output_dir="output")
charts = visualizer.generate_full_report(df, scores)
```

### Các biểu đồ được tạo

| File | Loại | Mô tả |
|------|------|-------|
| `metrics_summary.png` | Bar chart | Tổng hợp tất cả metrics |
| `radar_chart.png` | Radar | RAG Triad visualization |
| `correlation_heatmap.png` | Heatmap | Tương quan giữa metrics |
| `distribution_*.png` | Histogram | Phân phối từng metric |
| `metrics_by_type.png` | Grouped bar | So sánh theo loại câu hỏi |
| `latency_distribution.png` | Histogram | Thời gian phản hồi |

---

## 🔍 Giải thích kết quả

### Ma trận chẩn đoán (Diagnosis Matrix)

| Faithfulness | Relevancy | Chẩn đoán | Hành động |
|--------------|-----------|-----------|-----------|
| ✅ Cao | ✅ Cao | Hệ thống tốt | Tiếp tục giám sát |
| ❌ Thấp | ✅ Cao | **HALLUCINATION** | Giảm temperature, cải thiện prompt |
| ✅ Cao | ❌ Thấp | **EVASIVENESS** | Cải thiện retriever, prompt trực diện hơn |
| ❌ Thấp | ❌ Thấp | **SYSTEM FAILURE** | Kiểm tra toàn bộ pipeline |

### Ngưỡng điểm

| Score | Status | Màu |
|-------|--------|-----|
| ≥ 0.8 | Excellent | 🟢 |
| ≥ 0.6 | Good | 🔵 |
| ≥ 0.4 | Warning | 🟡 |
| < 0.4 | Critical | 🔴 |

---

## 📁 Cấu trúc thư mục

```
evaluation/
├── __init__.py              # Package exports
├── metrics.py               # 7 metrics implementations
├── testset_generator.py     # Synthetic data generation
├── evaluator.py             # Batch evaluation engine
├── visualizer.py            # Chart generation
├── evaluate.py              # Main entry point (menu)
├── test_questions.json      # 15 manual test cases
├── requirements_eval.txt    # Dependencies
├── README.md                # This file
└── output/                  # Generated results (auto-created)
    ├── *.csv                # Evaluation results
    ├── *.json               # Detailed results
    └── *.png                # Charts
```

---

## 🤝 Đóng góp

1. Thêm test cases vào `test_questions.json`
2. Mở rộng metrics trong `metrics.py`
3. Thêm biểu đồ mới trong `visualizer.py`

---

## 📚 Tài liệu tham khảo

- [Ragas Documentation](https://docs.ragas.io/)
- [DeepEval Documentation](https://docs.confident-ai.com/)
- [LangChain Evaluation](https://python.langchain.com/docs/guides/evaluation/)
