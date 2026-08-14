"""
ViFinQA Common Schemas
──────────────────────────────────────────────────────────────
Pydantic models for structured data throughout the pipeline.
"""

from typing import List, Dict, Optional, Any, Tuple
from pydantic import BaseModel, Field


# ── Pipeline 1 Schemas ──

class TableMetadata(BaseModel):
    """Metadata for a single extracted table (JSON Schema Store)."""
    doc_id: str = Field(..., description="Unique ID: TICKER_YEAR_TYPE_pN_lM")
    ticker: str
    year: int
    report_type: str = Field(..., description="consolidated | separate")
    page_number: int
    line_number: int
    category: str = Field(
        ..., description="balance_sheet | income_statement | cash_flow | notes"
    )
    title: str = Field(default="", description="Context/title from preceding text")
    unit: str = Field(default="VND", description="Detected unit: VND, triệu đồng, ...")
    columns: List[str] = Field(default_factory=list)
    num_rows: int = 0
    num_cols: int = 0
    source_file: str = ""


class TableSummary(BaseModel):
    """Template-based summary for a table (summaries.jsonl)."""
    doc_id: str
    summary: str
    ticker: str
    year: int
    report_type: str
    category: str
    key_terms: List[str] = Field(default_factory=list)


# ── Pipeline 2 Schemas ──

class Question(BaseModel):
    """A single question from the ViFinQA dataset."""
    id: int
    question: str


class ParsedQuestion(BaseModel):
    """Structured representation of a parsed financial question."""
    id: int
    question: str
    tickers: List[str] = Field(default_factory=list)
    years: List[int] = Field(default_factory=list)
    report_type: str = Field(
        default="", description="consolidated | separate | empty"
    )
    target_unit: str = Field(default="", description="Raw unit string from question")
    unit_type: str = Field(
        default="", description="absolute | percent | ratio | count | year"
    )
    complexity: str = Field(
        default="simple",
        description="simple | comparative | multi_company | time_series | conditional",
    )
    year_range: Optional[Tuple[int, int]] = None


class RetrievalResult(BaseModel):
    """Result of retrieving relevant data for a question."""
    question_id: int
    ticker: str
    year: int
    report_type: str
    report_path: Optional[str] = None
    relevant_doc_ids: List[str] = Field(default_factory=list)
    success: bool = False
    error: str = ""


class CodeExecutionResult(BaseModel):
    """Result of executing generated Pandas code."""
    code: str
    output: Optional[Any] = None
    success: bool = False
    error: str = ""
    retries: int = 0


class SubmissionEntry(BaseModel):
    """A single entry in the competition submission."""
    question_id: int
    answer: str
    pandas_code: str = ""
    source_doc_ids: List[str] = Field(default_factory=list)
