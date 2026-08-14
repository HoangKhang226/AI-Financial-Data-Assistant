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


def build_submission(results: List[dict]) -> dict:
    """Build submission dict from pipeline results.

    Args:
        results: List of dicts with 'question_id' and 'formatted_answer'.

    Returns:
        Dict mapping question_id (str) → answer (str).
    """
    submission = {}
    for r in results:
        qid = str(r.get("question_id", r.get("id", "")))
        answer = r.get("formatted_answer", r.get("answer", "0"))
        submission[qid] = str(answer) if answer else "0"
    return submission


def save_submission(
    submission: dict,
    output_path: str = "output/submission.json",
) -> str:
    """Save submission to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)
    logger.info(f"Submission saved: {output_path} ({len(submission)} answers)")
    return output_path
