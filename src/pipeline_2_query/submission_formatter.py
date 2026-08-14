"""
ViFinQA Submission Formatter
──────────────────────────────────────────────────────────────
Formats and packages answers into R2AI2026 submission JSON.

Refactored from: AIGure_S2/pipeline.py (generate_submission)
"""

import json
import os
from typing import List, Dict
from src.common.logger import get_logger

logger = get_logger(__name__)


def format_answer(value: float, unit_type: str) -> str:
    """Format a numeric answer to appropriate precision."""
    if unit_type in ("percent", "pct_point", "ratio"):
        return f"{value:.2f}"
    elif unit_type in ("count", "year"):
        return f"{int(value)}"
    else:
        if value == int(value):
            return f"{int(value)}"
        return f"{value:.2f}"


def build_submission(results: List[dict]) -> List[dict]:
    """Build submission JSON array from pipeline results."""
    submission = []
    for r in results:
        try:
            ans_val = float(r.get("answer", 0))
        except (ValueError, TypeError):
            ans_val = 0.0

        sub_item = {
            "id": r.get("id", int(r.get("question_id", 0)) if str(r.get("question_id", "0")).isdigit() else r.get("question_id")),
            "question": r.get("question", ""),
            "answer": ans_val,
            "relevant_docs": r.get("relevant_docs", []),
            "relevant_tables": r.get("relevant_tables", []),
            "evidence": r.get("evidence", []),
            "pandas_query": r.get("pandas_query", "")
        }
        submission.append(sub_item)
    return submission


def save_submission(
    submission: List[dict],
    output_path: str = "output/submission.json",
) -> str:
    """Save submission to JSON file and create submission.zip."""
    import zipfile
    import shutil
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)
    logger.info(f"Submission JSON saved: {output_path} ({len(submission)} answers)")

    # Create ZIP file
    zip_path = output_path.replace(".json", ".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Add submission.json to the root of the ZIP
        zf.write(output_path, arcname="submission.json")
        
        # 2. Add required CSVs into data/ folder inside ZIP
        added_csvs = set()
        for item in submission:
            for ev in item.get("evidence", []):
                csv_path_in_zip = ev.get("csv_path", "")
                if not csv_path_in_zip.startswith("data/"):
                    continue
                    
                csv_filename = os.path.basename(csv_path_in_zip)
                # Actual CSV is in data/csv_warehouse/
                actual_csv_path = os.path.join("data", "csv_warehouse", csv_filename)
                
                if actual_csv_path not in added_csvs and os.path.exists(actual_csv_path):
                    zf.write(actual_csv_path, arcname=f"data/{csv_filename}")
                    added_csvs.add(actual_csv_path)

    logger.info(f"Submission ZIP created: {zip_path} (packaged {len(added_csvs)} CSV files)")
    return zip_path
