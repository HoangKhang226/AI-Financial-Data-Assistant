"""
ViFinQA Page Splitter
──────────────────────────────────────────────────────────────
Reads OCR-extracted .txt files and splits by ===== PAGE N =====
markers, tracking line numbers for downstream metadata.

Refactored from: AIGure_S2/table_extractor.py (file reading logic)
"""

import os
import re
import glob
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from src.common.logger import get_logger

logger = get_logger(__name__)

# Page boundary pattern in OCR output
PAGE_PATTERN = re.compile(r"=====\s*PAGE\s+(\d+)\s*=====")


@dataclass
class PageContent:
    """Content of a single page from an OCR-extracted document."""
    page_number: int
    start_line: int       # 1-indexed line number in the original file
    end_line: int         # 1-indexed line number (inclusive)
    lines: List[str] = field(default_factory=list)
    has_tables: bool = False

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


@dataclass
class DocumentPages:
    """All pages from a single OCR-extracted document."""
    filepath: str
    ticker: str
    year: int
    report_type: str     # "consolidated" | "separate"
    pages: List[PageContent] = field(default_factory=list)
    total_lines: int = 0


def find_report_path(
    fs_dir: str,
    ticker: str,
    year: int,
    report_type: str = "",
) -> Optional[str]:
    """Find the report file path for a given ticker, year, and type.

    Args:
        fs_dir: Base directory containing financial_statements.
        ticker: Stock ticker code (e.g., "VJC").
        year: Fiscal year.
        report_type: "consolidated" | "separate" | "" (auto).

    Returns:
        Path to the *_extracted.txt file, or None.
    """
    base = os.path.join(fs_dir, ticker, str(year))
    if not os.path.exists(base):
        return None

    subdirs = [
        d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))
    ]

    if report_type == "separate":
        candidates = [d for d in subdirs if "separate" in d]
    elif report_type == "consolidated":
        candidates = [d for d in subdirs if "consolidated" in d]
    else:
        # Prefer consolidated, fallback to separate
        candidates = [d for d in subdirs if "consolidated" in d]
        if not candidates:
            candidates = [d for d in subdirs if "separate" in d]

    if not candidates:
        candidates = subdirs

    if not candidates:
        return None

    report_dir = os.path.join(base, candidates[0])
    txt_files = glob.glob(os.path.join(report_dir, "*extracted.txt"))
    return txt_files[0] if txt_files else None


def split_pages(filepath: str) -> List[PageContent]:
    """Split an OCR-extracted text file into pages.

    Args:
        filepath: Path to the *_extracted.txt file.

    Returns:
        List of PageContent objects, one per page.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    pages: List[PageContent] = []
    current_page: Optional[PageContent] = None

    for i, raw_line in enumerate(all_lines):
        line = raw_line.rstrip("\n\r")
        line_num = i + 1  # 1-indexed

        # Check for page boundary
        match = PAGE_PATTERN.match(line)
        if match:
            # Finalize previous page
            if current_page is not None:
                current_page.end_line = line_num - 1
                pages.append(current_page)

            # Start new page
            page_num = int(match.group(1))
            current_page = PageContent(
                page_number=page_num,
                start_line=line_num + 1,
                end_line=line_num + 1,
            )
            continue

        if current_page is not None:
            current_page.lines.append(line)
            if "<table" in line.lower():
                current_page.has_tables = True
        else:
            # Content before first page marker
            if current_page is None and line.strip():
                current_page = PageContent(
                    page_number=0,
                    start_line=1,
                    end_line=1,
                )
                current_page.lines.append(line)
                if "<table" in line.lower():
                    current_page.has_tables = True

    # Finalize last page
    if current_page is not None:
        current_page.end_line = len(all_lines)
        pages.append(current_page)

    return pages


def parse_document(
    filepath: str,
    ticker: str = "",
    year: int = 0,
    report_type: str = "",
) -> DocumentPages:
    """Parse a complete OCR document into pages.

    Args:
        filepath: Path to the extracted text file.
        ticker: Stock ticker (inferred from path if empty).
        year: Fiscal year (inferred from path if 0).
        report_type: Report type (inferred from path if empty).

    Returns:
        DocumentPages object with all parsed pages.
    """
    # Infer metadata from path if not provided
    if not ticker or not year or not report_type:
        parts = filepath.replace("\\", "/").split("/")
        for i, part in enumerate(parts):
            if part == "financial_statements" and i + 2 < len(parts):
                if not ticker:
                    ticker = parts[i + 1]
                if not year:
                    try:
                        year = int(parts[i + 2])
                    except ValueError:
                        pass
        if not report_type:
            if "consolidated" in filepath:
                report_type = "consolidated"
            elif "separate" in filepath:
                report_type = "separate"

    pages = split_pages(filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    doc = DocumentPages(
        filepath=filepath,
        ticker=ticker,
        year=year,
        report_type=report_type,
        pages=pages,
        total_lines=total_lines,
    )

    logger.info(
        f"Parsed document: {ticker}/{year}/{report_type} — "
        f"{len(pages)} pages, {total_lines} lines"
    )

    return doc
