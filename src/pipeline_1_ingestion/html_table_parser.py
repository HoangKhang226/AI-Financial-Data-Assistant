"""
ViFinQA HTML Table Parser
──────────────────────────────────────────────────────────────
BeautifulSoup-based parser for inline HTML <table> elements
in OCR-extracted financial reports. Handles rowspan/colspan
via a 2D matrix fill algorithm.

Refactored from: AIGure_S2/table_extractor.py (parsing logic)
"""

import re
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from src.common.logger import get_logger

logger = get_logger(__name__)

# ── Table Category Detection Keywords ──
TABLE_CATEGORIES = {
    "balance_sheet": [
        "cân đối kế toán", "bảng cân đối", "tài sản", "nguồn vốn",
    ],
    "income_statement": [
        "kết quả hoạt động kinh doanh", "kết quả kinh doanh",
        "báo cáo kết quả", "doanh thu bán hàng",
    ],
    "cash_flow": [
        "lưu chuyển tiền tệ", "lưu chuyển tiền", "cash flow",
    ],
    "notes": [
        "thuyết minh", "thuyết minh báo cáo tài chính",
    ],
}


@dataclass
class ParsedTable:
    """A single table parsed from HTML."""
    table_index: int         # Position in the document
    line_number: int         # Line number where table starts
    page_number: int         # Page containing this table
    category: str            # balance_sheet | income_statement | cash_flow | notes
    title: str               # Context/title from preceding text
    raw_html: str            # Original HTML string
    dataframe: Optional[pd.DataFrame] = None
    column_headers: List[str] = field(default_factory=list)
    num_rows: int = 0
    num_cols: int = 0


def extract_tables_from_lines(
    lines: List[str],
    page_number: int = 0,
    line_offset: int = 0,
) -> List[ParsedTable]:
    """Extract all HTML tables from a list of text lines.

    Args:
        lines: Lines of text (from a page or full document).
        page_number: Page number for metadata.
        line_offset: Offset to add to line numbers (for full-doc numbering).

    Returns:
        List of ParsedTable objects.
    """
    tables = []
    table_idx = 0

    for i, line in enumerate(lines):
        if "<table" not in line.lower():
            continue

        abs_line = i + line_offset + 1  # 1-indexed

        # Get context: preceding non-empty, non-HTML lines as title
        title = _extract_title(lines, i)

        # Classify the table category
        category = classify_table(line, title, lines, i)

        # Parse to DataFrame
        df = parse_table_html(line)

        table = ParsedTable(
            table_index=table_idx,
            line_number=abs_line,
            page_number=page_number,
            category=category,
            title=title[:200],
            raw_html=line,
            dataframe=df,
            column_headers=list(df.columns) if df is not None else [],
            num_rows=len(df) if df is not None else 0,
            num_cols=len(df.columns) if df is not None else 0,
        )
        tables.append(table)
        table_idx += 1

    return tables


def _extract_title(lines: List[str], table_line_idx: int) -> str:
    """Extract contextual title from lines preceding a table."""
    for j in range(table_line_idx - 1, max(0, table_line_idx - 15), -1):
        candidate = lines[j].strip()
        if candidate and "<" not in candidate and len(candidate) > 5:
            return candidate
    return ""


def classify_table(
    html: str,
    title: str,
    lines: List[str],
    line_idx: int,
) -> str:
    """Classify a table into a financial statement category.

    Checks surrounding context (title + 20 preceding lines) and
    HTML content for category keywords.
    """
    # Build context from title + preceding lines
    context = title.lower()
    for j in range(max(0, line_idx - 20), line_idx):
        context += " " + lines[j].lower()

    for category, keywords in TABLE_CATEGORIES.items():
        for kw in keywords:
            if kw in context:
                return category

    # Check the HTML content itself
    html_lower = html.lower()
    if "tài sản" in html_lower and "nguồn vốn" in html_lower:
        return "balance_sheet"
    if "doanh thu" in html_lower and "lợi nhuận" in html_lower:
        return "income_statement"
    if "lưu chuyển tiền" in html_lower or "tiền thuần" in html_lower:
        return "cash_flow"

    return "notes"


def parse_table_html(html: str) -> Optional[pd.DataFrame]:
    """Parse an HTML string into a clean Pandas DataFrame.

    Tries pd.read_html first, falls back to manual BeautifulSoup parsing.
    """
    try:
        dfs = pd.read_html(StringIO(html))
        if not dfs:
            return None

        df = dfs[0]
        df = _clean_dataframe(df)
        return df

    except Exception:
        # Fallback: manual parsing with BeautifulSoup
        try:
            return _manual_parse(html)
        except Exception:
            return None


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize a financial DataFrame."""
    # Drop rows where all values are NaN
    df = df.dropna(how="all")
    # Drop columns where all values are NaN
    df = df.dropna(axis=1, how="all")
    # Reset index
    df = df.reset_index(drop=True)
    return df


def _manual_parse(html: str) -> Optional[pd.DataFrame]:
    """Manually parse HTML table using BeautifulSoup with rowspan/colspan.

    Uses a 2D matrix fill algorithm to correctly expand merged cells.
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return None

    # First pass: determine grid dimensions
    trs = table.find_all("tr")
    if not trs:
        return None

    # Build 2D grid handling rowspan/colspan
    max_cols = 0
    for tr in trs:
        col_count = 0
        for cell in tr.find_all(["td", "th"]):
            colspan = int(cell.get("colspan", 1))
            col_count += colspan
        max_cols = max(max_cols, col_count)

    num_rows = len(trs)
    grid = [[None] * max_cols for _ in range(num_rows)]

    for row_idx, tr in enumerate(trs):
        col_idx = 0
        for cell in tr.find_all(["td", "th"]):
            # Skip filled cells
            while col_idx < max_cols and grid[row_idx][col_idx] is not None:
                col_idx += 1
            if col_idx >= max_cols:
                break

            text = cell.get_text(strip=True)
            rowspan = int(cell.get("rowspan", 1))
            colspan = int(cell.get("colspan", 1))

            # Fill the grid
            for dr in range(rowspan):
                for dc in range(colspan):
                    r = row_idx + dr
                    c = col_idx + dc
                    if r < num_rows and c < max_cols:
                        grid[r][c] = text

            col_idx += colspan

    # Replace None with empty string
    for r in range(num_rows):
        for c in range(max_cols):
            if grid[r][c] is None:
                grid[r][c] = ""

    if len(grid) < 2:
        return None

    df = pd.DataFrame(grid[1:], columns=grid[0])
    return df
