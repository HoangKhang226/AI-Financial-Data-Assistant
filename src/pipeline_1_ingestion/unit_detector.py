"""
ViFinQA Unit Detector
──────────────────────────────────────────────────────────────
Detects the original monetary unit declared in financial reports.
"""

import re
from typing import Optional

UNIT_PATTERNS = [
    (re.compile(r"đơn\s+vị\s*(?:tính)?\s*:\s*(tỷ\s+đồng)", re.I), "tỷ đồng"),
    (re.compile(r"đơn\s+vị\s*(?:tính)?\s*:\s*(triệu\s+đồng)", re.I), "triệu đồng"),
    (re.compile(r"đơn\s+vị\s*(?:tính)?\s*:\s*(nghìn\s+đồng)", re.I), "nghìn đồng"),
    (re.compile(r"đơn\s+vị\s*(?:tính)?\s*:\s*(đồng|VND)", re.I), "đồng"),
]

UNIT_MULTIPLIER = {
    "đồng": 1, "nghìn đồng": 1_000, "triệu đồng": 1_000_000,
    "tỷ đồng": 1_000_000_000, "nghìn tỷ đồng": 1_000_000_000_000,
}


def detect_unit(text: str) -> str:
    for pattern, unit in UNIT_PATTERNS:
        if pattern.search(text):
            return unit
    return "đồng"


def detect_unit_from_lines(lines: list, max_lines: int = 50) -> str:
    return detect_unit("\n".join(lines[:max_lines]))


def get_multiplier(unit: str) -> int:
    return UNIT_MULTIPLIER.get(unit, 1)
