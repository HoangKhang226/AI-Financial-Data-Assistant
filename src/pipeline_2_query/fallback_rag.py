"""
ViFinQA Fallback RAG
──────────────────────────────────────────────────────────────
Direct value extraction from DataFrames when code generation fails.
Deterministic regex-based search across table rows.

Refactored from: AIGure_S2/financial_calculator.py
"""

import re
import pandas as pd
from typing import List, Optional, Any
from src.pipeline_1_ingestion.ocr_normalizer import parse_numeric
from src.common.logger import get_logger

logger = get_logger(__name__)

# ── Row Label Patterns for Common Financial Items ──
ITEM_PATTERNS = {
    "tổng tài sản": [r"tổng\s*(cộng\s+)?tài\s+sản", r"total\s+assets"],
    "tài sản ngắn hạn": [r"tài\s+sản\s+ngắn\s+hạn"],
    "tài sản dài hạn": [r"tài\s+sản\s+dài\s+hạn"],
    "nợ ngắn hạn": [r"nợ\s+ngắn\s+hạn"],
    "nợ dài hạn": [r"nợ\s+dài\s+hạn"],
    "tổng nợ phải trả": [r"nợ\s+phải\s+trả"],
    "vốn chủ sở hữu": [r"vốn\s+chủ\s+sở\s+hữu"],
    "hàng tồn kho": [r"hàng\s+tồn\s+kho"],
    "tiền và các khoản tương đương tiền": [r"tiền\s+và\s+các\s+khoản\s+tương\s+đương\s+tiền"],
    "tiền mặt": [r"tiền\s+mặt", r"tiền\s+gửi\s+ngân\s+hàng"],
    "cho vay khách hàng": [r"cho\s+vay\s+khách\s+hàng"],
    "doanh thu thuần": [r"doanh\s+thu\s+thuần"],
    "doanh thu bán hàng": [r"doanh\s+thu\s+bán\s+hàng"],
    "giá vốn hàng bán": [r"giá\s+vốn\s+hàng\s+bán"],
    "lợi nhuận gộp": [r"lợi\s+nhuận\s+gộp"],
    "lợi nhuận trước thuế": [r"lợi\s+nhuận\s+(kế\s+toán\s+)?trước\s+thuế"],
    "lợi nhuận sau thuế": [r"lợi\s+nhuận\s+sau\s+thuế"],
    "chi phí tài chính": [r"chi\s+phí\s+tài\s+chính"],
    "chi phí dự phòng": [r"chi\s+phí\s+dự\s+phòng"],
    "thu nhập lãi thuần": [r"thu\s+nhập\s+lãi\s+thuần"],
    "lưu chuyển tiền thuần từ hoạt động kinh doanh": [
        r"lưu\s+chuyển\s+tiền\s+thuần\s+từ\s+.*hoạt\s+động\s*(?:kinh\s+doanh)"
    ],
    "lãi tiền gửi": [r"lãi\s+tiền\s+gửi", r"tiền\s+thu\s+lãi\s+tiền\s+gửi"],
}


def extract_value_from_df(
    df: pd.DataFrame,
    item_name: str,
    year_hint: Optional[str] = None,
) -> Optional[float]:
    """Find a financial item's value in a DataFrame by row label matching.

    Searches first 3 columns for label patterns, then extracts value
    from the appropriate numeric column.
    """
    if df is None or df.empty:
        return None

    item_lower = item_name.lower()
    patterns = ITEM_PATTERNS.get(item_lower, [re.escape(item_lower)])

    # Search for matching row
    matched_row = None
    for col_idx in range(min(3, len(df.columns))):
        col = df.columns[col_idx]
        col_series = df[col].astype(str).str.lower()
        for pattern in patterns:
            for idx, label in col_series.items():
                if re.search(pattern, label, re.IGNORECASE):
                    matched_row = idx
                    break
            if matched_row is not None:
                break
        if matched_row is not None:
            break

    # Fuzzy fallback
    if matched_row is None:
        for col_idx in range(min(3, len(df.columns))):
            col = df.columns[col_idx]
            for idx, val in df[col].astype(str).str.lower().items():
                if item_lower in val:
                    matched_row = idx
                    break
            if matched_row is not None:
                break

    if matched_row is None:
        return None

    # Find value column
    value_col = _find_value_column(df, year_hint)
    if value_col is None:
        return None

    raw = str(df.at[matched_row, value_col])
    return parse_numeric(raw)


def _find_value_column(
    df: pd.DataFrame, year_hint: Optional[str] = None,
) -> Optional[Any]:
    """Find the column containing target year's financial values."""
    if len(df.columns) <= 1:
        return None

    if year_hint:
        for col in df.columns:
            if year_hint in str(col).lower():
                return col

    # Heuristic: find rightmost numeric columns
    for col_idx in range(len(df.columns) - 1, 0, -1):
        col = df.columns[col_idx]
        start = min(2, len(df))
        sample = df[col].iloc[start:].dropna().astype(str)
        if len(sample) == 0:
            continue
        num_count = sum(
            1 for v in sample
            if re.match(r"^[\d.,()\\-]+$", v.strip()) and v.strip()
        )
        if num_count > len(sample) * 0.3:
            return col

    return df.columns[-2] if len(df.columns) >= 3 else df.columns[-1]


def extract_from_csv(
    csv_path: str, item_name: str, year: Optional[int] = None,
) -> Optional[float]:
    """Load a CSV and extract a value by item name."""
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        year_hint = str(year) if year else None
        return extract_value_from_df(df, item_name, year_hint)
    except Exception:
        return None
