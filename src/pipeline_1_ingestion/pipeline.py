"""
ViFinQA Pipeline 1 Orchestrator
──────────────────────────────────────────────────────────────
Coordinates the full ingestion pipeline:
  OCR Text → Page Split → HTML Parse → CSV Export
           → Metadata Build → Summary Gen → Index Build
"""

import os
import glob
from typing import Optional
from src.common.logger import get_logger
from src.pipeline_1_ingestion.page_splitter import (
    parse_document, find_report_path,
)
from src.pipeline_1_ingestion.html_table_parser import extract_tables_from_lines
from src.pipeline_1_ingestion.unit_detector import detect_unit_from_lines
from src.pipeline_1_ingestion.csv_exporter import export_all_tables
from src.pipeline_1_ingestion.metadata_builder import build_and_save_all
from src.pipeline_1_ingestion.summary_generator import generate_and_save

logger = get_logger(__name__)


def run_ingestion(
    input_dir: str,
    csv_output_dir: str = "data/csv_warehouse",
    metadata_output_dir: str = "data/metadata",
    summaries_path: str = "data/index/summaries.jsonl",
    verbose: bool = False,
) -> dict:
    """Run Pipeline 1 on all OCR text files in input_dir.

    Args:
        input_dir: Path to financial_statements/ directory.
        csv_output_dir: Where to write CSV files.
        metadata_output_dir: Where to write JSON metadata.
        summaries_path: Path for summaries.jsonl output.
        verbose: Print progress to stdout.

    Returns:
        Stats dict with counts of processed files/tables.
    """
    stats = {"files": 0, "tables": 0, "errors": 0}

    # Find all *_extracted.txt files
    pattern = os.path.join(input_dir, "**", "*_extracted.txt")
    txt_files = glob.glob(pattern, recursive=True)

    logger.info(f"Found {len(txt_files)} extracted text files")

    for filepath in txt_files:
        try:
            _process_single_file(
                filepath, csv_output_dir, metadata_output_dir,
                summaries_path, verbose,
            )
            stats["files"] += 1
        except Exception as e:
            stats["errors"] += 1
            logger.error(f"Error processing {filepath}: {e}")

    logger.info(
        f"Pipeline 1 complete: {stats['files']} files, "
        f"{stats['tables']} tables, {stats['errors']} errors"
    )
    
    # Step 7: Build Indices (Dense + Sparse)
    try:
        from src.pipeline_1_ingestion.index_builder import IndexBuilder
        logger.info("Building dense and sparse indices...")
        builder = IndexBuilder(
            dense_dir=os.path.join(os.path.dirname(summaries_path), "dense_vectors"),
            bm25_dir=os.path.join(os.path.dirname(summaries_path), "bm25_index")
        )
        # For now just build sparse index to test (dense requires downloading BAAI/bge-m3 which is large)
        builder.build_all(summaries_path, embedding_client=None)
        logger.info("Index building complete.")
    except Exception as e:
        logger.error(f"Failed to build indices: {e}")

    return stats


def _process_single_file(
    filepath: str,
    csv_output_dir: str,
    metadata_output_dir: str,
    summaries_path: str,
    verbose: bool,
) -> None:
    """Process a single OCR text file through the full Pipeline 1."""
    # Step 1: Parse document into pages
    doc = parse_document(filepath)

    if verbose:
        print(f"  [{doc.ticker}/{doc.year}/{doc.report_type}] "
              f"{len(doc.pages)} pages")

    # Step 2: Extract tables from each page
    all_tables = []
    for page in doc.pages:
        if not page.has_tables:
            continue
        tables = extract_tables_from_lines(
            page.lines, page.page_number, page.start_line - 1
        )
        all_tables.extend(tables)

    if not all_tables:
        return

    # Step 3: Detect unit
    first_page_lines = doc.pages[0].lines if doc.pages else []
    unit = detect_unit_from_lines(first_page_lines)

    # Step 4: Export CSVs
    export_all_tables(
        all_tables, doc.ticker, doc.year, doc.report_type, csv_output_dir
    )

    # Step 5: Build metadata
    metadata_list = build_and_save_all(
        all_tables, doc.ticker, doc.year, doc.report_type,
        unit, filepath, metadata_output_dir,
    )

    # Step 6: Generate summaries
    generate_and_save(metadata_list, summaries_path)

    if verbose:
        print(f"    -> {len(all_tables)} tables, unit={unit}")

if __name__ == "__main__":
    import argparse
    import sys
    
    # Ensure src is in the path
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
        
    parser = argparse.ArgumentParser(description="Run Pipeline 1 Ingestion")
    parser.add_argument("--input", type=str, default="data/ViFinQA/financial_statements", help="Input directory")
    parser.add_argument("--csv-out", type=str, default="data/csv_warehouse", help="CSV output directory")
    parser.add_argument("--meta-out", type=str, default="data/metadata", help="Metadata output directory")
    parser.add_argument("--summ-out", type=str, default="data/index/summaries.jsonl", help="Summaries output path")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    
    args = parser.parse_args()
    
    print("="*60)
    print("STARTING PIPELINE 1: INGESTION")
    print("="*60)
    
    run_ingestion(
        args.input, 
        args.csv_out, 
        args.meta_out, 
        args.summ_out, 
        args.verbose
    )
