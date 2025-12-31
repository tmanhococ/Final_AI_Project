"""
Synthetic Test Dataset Generator for RAG Chatbot Evaluation.

Generates test questions using different evolution strategies:
- Simple Evolution: Direct lookup questions
- Reasoning Evolution: Multi-step reasoning questions
- Multi-Context Evolution: Questions requiring info from multiple chunks

Author: AI Evaluation Framework
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

# Conditional import for Document type
if TYPE_CHECKING:
    from langchain_core.documents import Document

# Import shared rate limiter
from .rate_limiter import wait_for_rate_limit


@dataclass
class TestCase:
    """A single test case with question, contexts, and optional ground truth."""
    question: str
    ground_truth: str
    contexts: List[str] = field(default_factory=list)
    evolution_type: str = "simple"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "question": self.question,
            "ground_truth": self.ground_truth,
            "contexts": self.contexts,
            "evolution_type": self.evolution_type,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestCase":
        """Create TestCase from dictionary."""
        return cls(
            question=data["question"],
            ground_truth=data["ground_truth"],
            contexts=data.get("contexts", []),
            evolution_type=data.get("evolution_type", "simple"),
            metadata=data.get("metadata", {})
        )


class TestsetGenerator:
    """
    Generator for synthetic test datasets.
    
    Uses LLM to create diverse test questions from document corpus
    with different evolution strategies to test all chatbot flows.
    """
    
    def __init__(
        self, 
        llm=None, 
        documents: Optional[List["Document"]] = None,
        rate_limit_delay: float = 4.0  # Delay between API calls (seconds)
    ):
        """
        Initialize the testset generator.
        
        Args:
            llm: LangChain LLM instance (uses Gemini from config if None)
            documents: Optional list of documents to generate questions from
            rate_limit_delay: Delay in seconds between API calls (default: 4.0 for free Gemini API)
        """
        self._llm = llm
        self.documents = documents or []
        self.rate_limit_delay = rate_limit_delay
        # Note: Using shared rate limiter from rate_limiter.py
    
    def _get_llm(self):
        """Get or create LLM instance."""
        if self._llm is None:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from src.chatbot.config import CHATBOT_CONFIG
            
            self._llm = ChatGoogleGenerativeAI(
                model=CHATBOT_CONFIG.llm_model_name,
                google_api_key=CHATBOT_CONFIG.google_api_key,
                temperature=0.7  # Higher creativity for diverse questions
            )
        return self._llm
    
    def load_documents_from_directory(self, directory: str | Path) -> List[Document]:
        """
        Load documents from a directory.
        
        Args:
            directory: Path to directory containing text files
            
        Returns:
            List of loaded documents
        """
        from langchain_community.document_loaders import TextLoader, DirectoryLoader
        
        directory = Path(directory)
        
        if not directory.exists():
            raise ValueError(f"Directory not found: {directory}")
        
        loader = DirectoryLoader(
            str(directory),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        
        self.documents = loader.load()
        return self.documents
    
    def _chunk_documents(self, chunk_size: int = 500) -> List[str]:
        """
        Split documents into smaller chunks.
        
        Args:
            chunk_size: Approximate size of each chunk
            
        Returns:
            List of text chunks
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=50
        )
        
        chunks = []
        for doc in self.documents:
            splits = splitter.split_text(doc.page_content)
            chunks.extend(splits)
        
        return chunks
    
    def _generate_simple_question(self, chunk: str) -> Optional[TestCase]:
        """
        Generate a simple lookup question from a chunk.
        
        Simple questions: Direct fact lookup
        Example: "Thủ đô của Việt Nam là gì?"
        
        Args:
            chunk: Text chunk to generate question from
            
        Returns:
            TestCase with simple question
        """
        llm = self._get_llm()
        
        prompt = f"""Bạn là chuyên gia tạo câu hỏi kiểm thử cho chatbot sức khỏe.
Dựa vào đoạn văn sau, hãy tạo MỘT câu hỏi tra cứu đơn giản mà câu trả lời có thể tìm thấy trực tiếp trong đoạn văn.

Đoạn văn:
\"\"\"
{chunk}
\"\"\"

Trả về theo định dạng:
CÂU HỎI: [câu hỏi của bạn]
CÂU TRẢ LỜI: [câu trả lời dựa trên đoạn văn]"""
        
        try:
            wait_for_rate_limit()
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse response
            lines = content.strip().split('\n')
            question = ""
            answer = ""
            
            for line in lines:
                if line.startswith("CÂU HỎI:"):
                    question = line.replace("CÂU HỎI:", "").strip()
                elif line.startswith("CÂU TRẢ LỜI:"):
                    answer = line.replace("CÂU TRẢ LỜI:", "").strip()
            
            if question and answer:
                return TestCase(
                    question=question,
                    ground_truth=answer,
                    contexts=[chunk],
                    evolution_type="simple"
                )
                
        except Exception as e:
            print(f"Error generating simple question: {e}")
        
        return None
    
    def _generate_reasoning_question(self, chunk: str) -> Optional[TestCase]:
        """
        Generate a reasoning question requiring multi-step thinking.
        
        Reasoning questions: Require inference and analysis
        Example: "Dựa vào vị trí địa lý của Hà Nội, hãy giải thích vai trò kinh tế của nó."
        
        Args:
            chunk: Text chunk to generate question from
            
        Returns:
            TestCase with reasoning question
        """
        llm = self._get_llm()
        
        prompt = f"""Bạn là chuyên gia tạo câu hỏi kiểm thử cho chatbot sức khỏe.
Dựa vào đoạn văn sau, hãy tạo MỘT câu hỏi YÊU CẦU SUY LUẬN nhiều bước.
Câu hỏi nên bắt đầu bằng "Giải thích...", "Tại sao...", "Phân tích...", "So sánh..." hoặc tương tự.

Đoạn văn:
\"\"\"
{chunk}
\"\"\"

Trả về theo định dạng:
CÂU HỎI: [câu hỏi suy luận của bạn]
CÂU TRẢ LỜI: [câu trả lời chi tiết dựa trên đoạn văn]"""
        
        try:
            wait_for_rate_limit()
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse response
            lines = content.strip().split('\n')
            question = ""
            answer = ""
            
            current_field = None
            for line in lines:
                if line.startswith("CÂU HỎI:"):
                    current_field = "question"
                    question = line.replace("CÂU HỎI:", "").strip()
                elif line.startswith("CÂU TRẢ LỜI:"):
                    current_field = "answer"
                    answer = line.replace("CÂU TRẢ LỜI:", "").strip()
                elif current_field == "answer":
                    answer += " " + line.strip()
            
            if question and answer:
                return TestCase(
                    question=question,
                    ground_truth=answer.strip(),
                    contexts=[chunk],
                    evolution_type="reasoning"
                )
                
        except Exception as e:
            print(f"Error generating reasoning question: {e}")
        
        return None
    
    def _generate_multi_context_question(
        self, 
        chunk1: str, 
        chunk2: str
    ) -> Optional[TestCase]:
        """
        Generate a question requiring information from multiple chunks.
        
        Multi-context questions: Require synthesis from 2+ sources
        Example: "So sánh chính sách hoàn tiền của công ty A và công ty B."
        
        Args:
            chunk1: First text chunk
            chunk2: Second text chunk
            
        Returns:
            TestCase with multi-context question
        """
        llm = self._get_llm()
        
        prompt = f"""Bạn là chuyên gia tạo câu hỏi kiểm thử cho chatbot sức khỏe.
Dựa vào HAI đoạn văn dưới đây, hãy tạo MỘT câu hỏi mà người dùng cần KẾT HỢP thông tin từ cả hai đoạn để trả lời.
Câu hỏi có thể yêu cầu so sánh, tổng hợp, hoặc liên kết các khái niệm.

ĐOẠN VĂN 1:
\"\"\"
{chunk1}
\"\"\"

ĐOẠN VĂN 2:
\"\"\"
{chunk2}
\"\"\"

Trả về theo định dạng:
CÂU HỎI: [câu hỏi cần kết hợp thông tin]
CÂU TRẢ LỜI: [câu trả lời tổng hợp từ cả hai đoạn]"""
        
        try:
            wait_for_rate_limit()
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse response
            lines = content.strip().split('\n')
            question = ""
            answer = ""
            
            current_field = None
            for line in lines:
                if line.startswith("CÂU HỎI:"):
                    current_field = "question"
                    question = line.replace("CÂU HỎI:", "").strip()
                elif line.startswith("CÂU TRẢ LỜI:"):
                    current_field = "answer"
                    answer = line.replace("CÂU TRẢ LỜI:", "").strip()
                elif current_field == "answer":
                    answer += " " + line.strip()
            
            if question and answer:
                return TestCase(
                    question=question,
                    ground_truth=answer.strip(),
                    contexts=[chunk1, chunk2],
                    evolution_type="multi_context"
                )
                
        except Exception as e:
            print(f"Error generating multi-context question: {e}")
        
        return None
    
    def generate(
        self,
        test_size: int = 20,
        distribution: Optional[Dict[str, float]] = None,
        verbose: bool = True
    ) -> List[TestCase]:
        """
        Generate a synthetic test dataset.
        
        Args:
            test_size: Number of test cases to generate
            distribution: Distribution of question types
                         Default: {"simple": 0.5, "reasoning": 0.3, "multi_context": 0.2}
            verbose: Print progress messages
            
        Returns:
            List of TestCase objects
        """
        if not self.documents:
            raise ValueError("No documents loaded. Call load_documents_from_directory first.")
        
        # Default distribution
        if distribution is None:
            distribution = {
                "simple": 0.5,
                "reasoning": 0.3,
                "multi_context": 0.2
            }
        
        # Chunk documents
        chunks = self._chunk_documents()
        
        if verbose:
            print(f"Loaded {len(self.documents)} documents, split into {len(chunks)} chunks")
        
        if len(chunks) < 2:
            raise ValueError("Need at least 2 chunks to generate questions")
        
        # Calculate number of each type
        n_simple = int(test_size * distribution.get("simple", 0.5))
        n_reasoning = int(test_size * distribution.get("reasoning", 0.3))
        n_multi = test_size - n_simple - n_reasoning
        
        test_cases: List[TestCase] = []
        
        # Generate simple questions
        if verbose:
            print(f"\nGenerating {n_simple} simple questions...")
        
        random.shuffle(chunks)
        for i, chunk in enumerate(chunks[:n_simple * 2]):  # Try more chunks in case some fail
            if len([t for t in test_cases if t.evolution_type == "simple"]) >= n_simple:
                break
            
            case = self._generate_simple_question(chunk)
            if case:
                test_cases.append(case)
                if verbose:
                    print(f"  [{len(test_cases)}/{test_size}] Simple: {case.question[:50]}...")
        
        # Generate reasoning questions
        if verbose:
            print(f"\nGenerating {n_reasoning} reasoning questions...")
        
        random.shuffle(chunks)
        for i, chunk in enumerate(chunks[:n_reasoning * 2]):
            if len([t for t in test_cases if t.evolution_type == "reasoning"]) >= n_reasoning:
                break
            
            case = self._generate_reasoning_question(chunk)
            if case:
                test_cases.append(case)
                if verbose:
                    print(f"  [{len(test_cases)}/{test_size}] Reasoning: {case.question[:50]}...")
        
        # Generate multi-context questions
        if verbose:
            print(f"\nGenerating {n_multi} multi-context questions...")
        
        for i in range(n_multi * 2):
            if len([t for t in test_cases if t.evolution_type == "multi_context"]) >= n_multi:
                break
            
            # Pick two random chunks
            if len(chunks) >= 2:
                chunk1, chunk2 = random.sample(chunks, 2)
                case = self._generate_multi_context_question(chunk1, chunk2)
                if case:
                    test_cases.append(case)
                    if verbose:
                        print(f"  [{len(test_cases)}/{test_size}] Multi-context: {case.question[:50]}...")
        
        if verbose:
            print(f"\nGenerated {len(test_cases)} test cases total")
            print(f"  - Simple: {len([t for t in test_cases if t.evolution_type == 'simple'])}")
            print(f"  - Reasoning: {len([t for t in test_cases if t.evolution_type == 'reasoning'])}")
            print(f"  - Multi-context: {len([t for t in test_cases if t.evolution_type == 'multi_context'])}")
        
        return test_cases
    
    def save_testset(
        self,
        test_cases: List[TestCase],
        output_path: str | Path
    ) -> None:
        """
        Save test cases to a JSON file.
        
        Args:
            test_cases: List of TestCase objects
            output_path: Path to output JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "generated_at": datetime.now().isoformat(),
            "total_cases": len(test_cases),
            "distribution": {
                "simple": len([t for t in test_cases if t.evolution_type == "simple"]),
                "reasoning": len([t for t in test_cases if t.evolution_type == "reasoning"]),
                "multi_context": len([t for t in test_cases if t.evolution_type == "multi_context"])
            },
            "test_cases": [tc.to_dict() for tc in test_cases]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(test_cases)} test cases to {output_path}")
    
    @staticmethod
    def load_testset(input_path: str | Path) -> List[TestCase]:
        """
        Load test cases from a JSON file.
        
        Args:
            input_path: Path to input JSON file
            
        Returns:
            List of TestCase objects
        """
        input_path = Path(input_path)
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return [TestCase.from_dict(tc) for tc in data["test_cases"]]


def create_manual_testset() -> List[TestCase]:
    """
    Create a manual test set for immediate testing.
    
    Returns hardcoded test cases covering different chatbot flows:
    - Social flow: Greetings, who are you
    - Health + CSV: Statistics questions
    - Health + Retriever: Medical knowledge
    - Health + Both: Combined questions
    """
    test_cases = [
        # === Social Flow ===
        TestCase(
            question="Xin chào",
            ground_truth="Xin chào! Tôi là trợ lý sức khỏe AI. Tôi có thể giúp gì cho bạn?",
            evolution_type="social"
        ),
        TestCase(
            question="Bạn là ai?",
            ground_truth="Tôi là trợ lý sức khỏe AI, được thiết kế để hỗ trợ theo dõi và tư vấn sức khỏe khi sử dụng máy tính.",
            evolution_type="social"
        ),
        
        # === Health + CSV (Realtime Data) ===
        TestCase(
            question="Phân tích dữ liệu log sức khỏe của tôi",
            ground_truth="Dựa trên dữ liệu log, bạn có tổng cộng nhiều phiên làm việc với các chỉ số EAR, khoảng cách, và số lần buồn ngủ được ghi nhận.",
            evolution_type="csv"
        ),
        TestCase(
            question="Thời gian trung bình của các phiên làm việc là bao nhiêu?",
            ground_truth="Thời gian trung bình của các phiên làm việc được tính từ dữ liệu log.",
            evolution_type="csv"
        ),
        TestCase(
            question="Tôi có bao nhiêu lần buồn ngủ trong các phiên đo?",
            ground_truth="Số lần buồn ngủ được thống kê từ cột drowsiness_events trong dữ liệu log.",
            evolution_type="csv"
        ),
        
        # === Health + Retriever (Chunked Data) ===
        TestCase(
            question="Hội chứng thị giác máy tính (CVS) là gì?",
            ground_truth="Hội chứng thị giác máy tính (Computer Vision Syndrome - CVS) là tình trạng mỏi mắt và các triệu chứng liên quan do sử dụng máy tính, điện thoại hoặc các thiết bị điện tử trong thời gian dài.",
            contexts=["CVS là hội chứng thị giác máy tính gây mỏi mắt khi làm việc với màn hình."],
            evolution_type="retriever"
        ),
        TestCase(
            question="Triệu chứng mỏi mắt khi làm việc máy tính là gì?",
            ground_truth="Triệu chứng mỏi mắt bao gồm: nhức mắt, khô mắt, nhìn mờ, đau đầu, và khó tập trung sau thời gian dài nhìn màn hình.",
            evolution_type="retriever"
        ),
        TestCase(
            question="Làm sao để phòng ngừa mỏi mắt khi sử dụng máy tính?",
            ground_truth="Để phòng ngừa mỏi mắt: áp dụng quy tắc 20-20-20 (mỗi 20 phút nhìn xa 20 feet trong 20 giây), điều chỉnh độ sáng màn hình, giữ khoảng cách phù hợp, và chớp mắt thường xuyên.",
            evolution_type="retriever"
        ),
        
        # === Health + Both (CSV + Retriever) ===
        TestCase(
            question="Dựa vào log của tôi, tôi có nguy cơ mắc hội chứng mỏi mắt không?",
            ground_truth="Dựa vào dữ liệu log và kiến thức y khoa về hội chứng thị giác máy tính, có thể đánh giá nguy cơ mỏi mắt của bạn.",
            evolution_type="both"
        ),
        TestCase(
            question="Phân tích chỉ số EAR của tôi và giải thích ý nghĩa của nó",
            ground_truth="Chỉ số EAR (Eye Aspect Ratio) đo mức độ mở mắt. Giá trị EAR thấp cho thấy mắt đang nhắm hoặc buồn ngủ. Từ dữ liệu log sẽ phân tích EAR trung bình của bạn.",
            evolution_type="both"
        ),
    ]
    
    return test_cases
