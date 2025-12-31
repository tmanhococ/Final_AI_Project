# RAG Chatbot Evaluation Results

**Ngày chạy:** 2025-12-31  
**Framework:** RAG Chatbot Evaluation Framework  
**Dataset:** Manual Test Set (10 câu hỏi)

---

## 📊 Tổng quan

| Nhóm Metrics | Số Metrics | Yêu cầu API | Trạng thái |
|--------------|------------|-------------|------------|
| Traditional | 3 (BLEU, ROUGE, BERTScore) | ❌ Không | ✅ Hoàn thành |
| RAG-Specific | 4 (Faithfulness, Relevancy, Precision, Recall) | ✅ Cần Gemini API | ⚠️ Bị rate limit |

---

## 📈 Option 3: BLEU Score

**Mô tả:** Đo độ trùng khớp n-gram giữa câu trả lời và ground truth.

| Evolution Type | Số mẫu | BLEU Avg |
|----------------|--------|----------|
| social | 2 | 0.0174 |
| csv | 3 | 0.0157 |
| retriever | 3 | 0.0041 |
| both | 2 | 0.0052 |
| **Tổng** | **10** | **0.0105** |

> **Nhận xét:** Điểm BLEU thấp là bình thường khi so sánh câu trả lời thực tế với ground truth do khác biệt về từ vựng. BLEU phù hợp cho dịch máy hơn là đánh giá RAG.

---

## 📈 Option 4: ROUGE Score

**Mô tả:** Đo độ phủ recall của n-gram từ ground truth trong câu trả lời.

| Evolution Type | Số mẫu | ROUGE Avg |
|----------------|--------|-----------|
| social | 2 | 0.0722 |
| csv | 3 | 0.0643 |
| retriever | 3 | 0.0488 |
| both | 2 | 0.0579 |
| **Tổng** | **10** | **0.0600** |

> **Nhận xét:** ROUGE cao hơn BLEU một chút nhưng vẫn thấp do focus vào recall. Cả hai metrics truyền thống đều có hạn chế cho RAG systems.

---

## 📈 Option 5: BERTScore

**Mô tả:** Đo độ tương đồng ngữ nghĩa sử dụng BERT embeddings (bert-base-multilingual-cased).

| Evolution Type | Số mẫu | BERTScore F1 |
|----------------|--------|--------------|
| social | 2 | 0.6596 |
| csv | 3 | 0.6212 |
| retriever | 3 | 0.6202 |
| both | 2 | 0.6270 |
| **Tổng** | **10** | **0.6297** |

**Chi tiết:**
| Metric | Score |
|--------|-------|
| Precision | 0.6472 |
| Recall | 0.6135 |
| F1 | 0.6297 |

> **Nhận xét:** BERTScore cao hơn đáng kể so với BLEU/ROUGE vì đo **semantic similarity** thay vì lexical overlap. Score ~0.63 cho thấy câu trả lời có ngữ nghĩa tương đối gần với ground truth mặc dù từ vựng khác.

---

## 📈 Option 6-9: RAG Metrics (LLM-as-Judge)

### ⚠️ Rate Limit Issue

Do sử dụng **Gemini Free Tier** với giới hạn:
- **5 requests/phút**
- **20 requests/ngày**

Các RAG metrics yêu cầu nhiều API calls:
- **Faithfulness:** 2+ calls/question (extract claims + verify each)
- **Answer Relevancy:** 2+ calls/question (generate questions + embed)
- **Context Precision:** 1+ calls/context chunk
- **Context Recall:** 2+ calls/question

### Giải pháp đề xuất:

1. **Upgrade Gemini API** - Pay-as-you-go hoặc Pro tier
2. **Rate limit delay = 45s** - Đã cấu hình trong `rate_limiter.py`
3. **Chạy từng metric riêng lẻ** - Để không vượt quota

---

## 📊 Kết quả tổng hợp

### Traditional Metrics (Đã chạy)

| Metric | Average Score | Status |
|--------|---------------|--------|
| BLEU | 0.0105 | ✅ |
| ROUGE | 0.0600 | ✅ |
| BERTScore | 0.6297 | ✅ |

### RAG Metrics (LLM-as-Judge)

| Metric | Expected Range | Status |
|--------|----------------|--------|
| Faithfulness | 0.0 - 1.0 | ⚠️ Rate limited |
| Answer Relevancy | 0.0 - 1.0 | ⚠️ Rate limited |
| Context Precision | 0.0 - 1.0 | ⚠️ Rate limited |
| Context Recall | 0.0 - 1.0 | ⚠️ Rate limited |

---

## 🔍 Phân tích

### Tại sao Traditional Metrics thấp?

1. **BLEU/ROUGE đo lexical overlap** - Không phù hợp cho RAG
2. **Câu trả lời có thể đúng nhưng diễn đạt khác** - Penalizes paraphrasing
3. **Ground truth chỉ là tham khảo** - Không phải unique answer

### RAG Metrics quan trọng hơn vì:

1. **Faithfulness** - Kiểm tra hallucination (bot có bịa không?)
2. **Answer Relevancy** - Câu trả lời có đúng trọng tâm không?
3. **Context Precision** - Retriever có lấy đúng context không?
4. **Context Recall** - Có bỏ sót thông tin quan trọng không?

---

## 📁 Output Files

| File | Mô tả |
|------|-------|
| `output/bleu_results.csv` | Chi tiết BLEU scores |
| `output/rouge_results.csv` | Chi tiết ROUGE scores |
| `output/metrics_summary.png` | Biểu đồ tổng hợp |
| `golden_dataset.json` | Test cases được sinh |

---

## 🚀 Hướng dẫn chạy tiếp

```bash
# 1. Chạy với Manual Test Set (không cần sinh dữ liệu)
python evaluate.py
# Chọn option 2 -> option 10 (All Traditional)

# 2. Với RAG metrics, chờ reset quota hoặc upgrade API
# Chọn option 11 (All RAG Metrics)
```

---

## 📋 Kết luận

- ✅ **Framework hoạt động tốt** - Tất cả modules đã được implement
- ✅ **Traditional Metrics chạy được** - Không cần API
- ⚠️ **RAG Metrics cần Gemini API** - Bị giới hạn free tier
- 💡 **Khuyến nghị:** Upgrade Gemini API để chạy full evaluation
