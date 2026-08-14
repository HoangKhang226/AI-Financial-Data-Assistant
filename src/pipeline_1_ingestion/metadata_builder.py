"""
ViFinQA Metadata Builder
──────────────────────────────────────────────────────────────
Generates JSON metadata for each extracted table.
Output: data/metadata/TICKER_YEAR_TYPE_pN_lM.json
"""

import os
import json
from typing import List, Optional
from src.pipeline_1_ingestion.html_table_parser import ParsedTable
from src.pipeline_1_ingestion.csv_exporter import generate_doc_id
from src.pipeline_1_ingestion.unit_detector import detect_unit_from_lines


def build_metadata(
    table: ParsedTable,
    ticker: str,
    year: int,
    report_type: str,
    unit: str = "đồng",
    source_file: str = "",
) -> dict:
    doc_id = generate_doc_id(
        ticker, year, report_type, table.page_number, table.line_number
    )
    return {
        "doc_id": doc_id,
        "ticker": ticker,
        "year": year,
        "report_type": report_type,
        "page_number": table.page_number,
        "line_number": table.line_number,
        "category": table.category,
        "title": table.title,
        "unit": unit,
        "columns": table.column_headers,
        "num_rows": table.num_rows,
        "num_cols": table.num_cols,
        "source_file": source_file,
    }


def save_metadata(metadata: dict, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{metadata['doc_id']}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return filepath


def build_and_save_all(
    tables: List[ParsedTable],
    ticker: str,
    year: int,
    report_type: str,
    unit: str,
    source_file: str,
    output_dir: str,
) -> List[dict]:
    all_meta = []
    for table in tables:
        meta = build_metadata(
            table, ticker, year, report_type, unit, source_file
        )
        save_metadata(meta, output_dir)
        all_meta.append(meta)
    return all_meta
