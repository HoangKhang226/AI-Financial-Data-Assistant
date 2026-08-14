"""
ViFinQA OCR Normalizer
──────────────────────────────────────────────────────────────
Fixes common OCR errors in Vietnamese financial documents:
  - Negative numbers in parentheses: (1.234) → -1234
  - Character confusion: O→0, l→1
  - Vietnamese number separators: dots as thousands, comma as decimal
  - Whitespace normalization

Refactored from: AIGure_S2/table_extractor.py (parse_numeric)
"""

import re
from typing import Optional


def parse_numeric(value: str) -> Optional[float]:
    """Parse a Vietnamese-formatted number string to float.

    Handles:
        "16.727.030.230.311" → 16727030230311.0
        "(174.500.000.000)" → -174500000000.0
        "1.234,56" → 1234.56
        "-" → 0.0

    Args:
        value: Raw string from OCR output.

    Returns:
        Parsed float value, or None if unparseable.
    """
    if not isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    s = value.strip()
    if not s or s == "-" or s == "–":
        return 0.0

    # Handle parenthesized negatives: (174.500.000.000) → negative
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1].strip()

    # Remove any non-numeric chars except . , -
    s = re.sub(r"[^\d.,-]", "", s)

    if not s:
        return None

    # Vietnamese format: dots as thousands, comma as decimal
    if "," in s:
        # "1.234.567,89" → "1234567.89"
        s = s.replace(".", "").replace(",", ".")
    else:
        # "16.727.030.230.311" → all dots are thousands separators
        # Heuristic: if more than one dot, they're thousands separators
        dot_count = s.count(".")
        if dot_count > 1:
            s = s.replace(".", "")
        elif dot_count == 1:
            # Could be thousands or decimal - check position
            parts = s.split(".")
            if len(parts[1]) == 3:
                # Likely thousands separator (1.234 = 1234)
                s = s.replace(".", "")
            # else keep as decimal

    try:
        result = float(s)
        return -result if negative else result
    except ValueError:
        return None


def fix_ocr_digits(text: str) -> str:
    """Fix common OCR character confusion in numeric contexts.

    Common OCR errors:
        - 'O' (letter O) → '0' (zero) in numeric contexts
        - 'l' (letter l) → '1' (one) in numeric contexts
        - 'S' → '5', 'B' → '8' in numeric contexts

    Args:
        text: Raw text from OCR.

    Returns:
        Text with OCR digit errors corrected.
    """
    # Only fix within numeric-looking sequences
    def _fix_match(m: re.Match) -> str:
        s = m.group(0)
        s = s.replace("O", "0").replace("o", "0")
        s = s.replace("l", "1").replace("I", "1")
        s = s.replace("S", "5").replace("B", "8")
        return s

    # Match sequences that look mostly numeric but have OCR errors
    pattern = r"[\d.,()OolISB]{4,}"
    return re.sub(pattern, _fix_match, text)


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in extracted text.

    - Collapse multiple spaces to single space
    - Strip leading/trailing whitespace per line
    - Preserve meaningful line breaks
    """
    lines = text.split("\n")
    normalized = []
    for line in lines:
        line = re.sub(r"[ \t]+", " ", line).strip()
        normalized.append(line)
    return "\n".join(normalized)


def normalize_number_separators(text: str) -> str:
    """Standardize number separator formats in text.

    Converts mixed formats to consistent Vietnamese format
    (dots as thousands, comma as decimal).
    """
    # Fix spaces used as thousands separators: "1 234 567" → "1.234.567"
    text = re.sub(
        r"(\d)\s+(\d{3})(?=\s+\d{3}|\b)",
        r"\1.\2",
        text,
    )
    return text
