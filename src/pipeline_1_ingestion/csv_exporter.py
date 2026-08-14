"""
ViFinQA CSV Exporter
──────────────────────────────────────────────────────────────
Exports parsed DataFrames to CSV files in data/csv_warehouse/.
Naming: TICKER_YEAR_TYPE_pN_lM.csv
"""

import os
import pandas as pd
from typing import Optional
from src.pipeline_1_ingestion.html_table_parser import ParsedTable


def generate_doc_id(
    ticker: str, year: int, report_type: str,
    page_number: int, line_number: int,
) -> str:
    return f"{ticker}_{year}_{report_type}_p{page_number}_l{line_number}"


def export_table(
    table: ParsedTable, ticker: str, year: int, report_type: str,
    output_dir: str,
) -> Optional[str]:
    if table.dataframe is None or table.dataframe.empty:
        return None

    doc_id = generate_doc_id(
        ticker, year, report_type, table.page_number, table.line_number
    )
    filename = f"{doc_id}.csv"
    filepath = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    table.dataframe.to_csv(filepath, index=False, encoding="utf-8-sig")
    return filepath


def export_all_tables(
    tables: list, ticker: str, year: int, report_type: str,
    output_dir: str,
) -> list:
    paths = []
    for table in tables:
        path = export_table(table, ticker, year, report_type, output_dir)
        if path:
            paths.append(path)
    return paths
