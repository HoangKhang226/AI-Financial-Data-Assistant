"""
ViFinQA Summary Generator
──────────────────────────────────────────────────────────────
Template-based summary generation (no LLM required).
Generates searchable text summaries from table metadata.
Output: data/index/summaries.jsonl
"""

import json
import os
from typing import List


CATEGORY_VI = {
    "balance_sheet": "Bảng cân đối kế toán",
    "income_statement": "Báo cáo kết quả hoạt động kinh doanh",
    "cash_flow": "Báo cáo lưu chuyển tiền tệ",
    "notes": "Thuyết minh báo cáo tài chính",
}

TYPE_VI = {
    "consolidated": "hợp nhất",
    "separate": "riêng (công ty mẹ)",
}


def generate_summary(metadata: dict) -> dict:
    ticker = metadata["ticker"]
    year = metadata["year"]
    rtype = TYPE_VI.get(metadata["report_type"], metadata["report_type"])
    category = CATEGORY_VI.get(metadata["category"], metadata["category"])
    title = metadata.get("title", "")
    unit = metadata.get("unit", "đồng")
    columns = metadata.get("columns", [])

    summary_text = (
        f"{category} {rtype} của {ticker} năm {year}. "
        f"Đơn vị: {unit}. "
    )
    if title:
        summary_text += f"Nội dung: {title}. "
    if columns:
        col_text = ", ".join(str(c) for c in columns[:8])
        summary_text += f"Các cột: {col_text}."

    key_terms = [ticker, str(year), metadata["report_type"], metadata["category"]]
    if title:
        key_terms.extend(title.split()[:10])

    return {
        "doc_id": metadata["doc_id"],
        "summary": summary_text,
        "ticker": ticker,
        "year": year,
        "report_type": metadata["report_type"],
        "category": metadata["category"],
        "key_terms": key_terms,
    }


def generate_and_save(
    metadata_list: List[dict],
    output_path: str,
) -> List[dict]:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summaries = []
    with open(output_path, "a", encoding="utf-8") as f:
        for meta in metadata_list:
            summary = generate_summary(meta)
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
            summaries.append(summary)
    return summaries
