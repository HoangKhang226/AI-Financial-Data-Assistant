"""
ViFinQA Unit Converter
──────────────────────────────────────────────────────────────
Matrix conversion between VND ↔ triệu ↔ tỷ ↔ nghìn tỷ.

Refactored from: AIGure_S2/config.py UNIT_MAP + financial_calculator.py
"""

from typing import Optional

UNIT_MULTIPLIER = {
    "đồng": 1,
    "nghìn đồng": 1_000,
    "triệu đồng": 1_000_000,
    "tỷ đồng": 1_000_000_000,
    "trăm tỷ đồng": 100_000_000_000,
    "nghìn tỷ đồng": 1_000_000_000_000,
}


def convert_unit(
    value: float,
    source_unit: str,
    target_unit: str,
) -> float:
    """Convert a value between Vietnamese monetary units.

    Args:
        value: Numeric value in source_unit.
        source_unit: e.g., "đồng", "triệu đồng".
        target_unit: e.g., "tỷ đồng".

    Returns:
        Converted value in target_unit.
    """
    src_mult = UNIT_MULTIPLIER.get(source_unit, 1)
    tgt_mult = UNIT_MULTIPLIER.get(target_unit, 1)

    if tgt_mult == 0:
        return value

    # Convert to VND base, then to target
    vnd_value = value * src_mult
    return vnd_value / tgt_mult


def convert_to_target(
    value: float,
    target_unit: str,
    unit_type: str,
) -> float:
    """Convert raw VND value to target unit (for submission).

    Only converts for "absolute" unit types.
    Percent/ratio/count values are returned as-is.
    """
    if unit_type == "absolute":
        divisor = UNIT_MULTIPLIER.get(target_unit, 1)
        if divisor > 0:
            return value / divisor
    return value


def compute_ratio(
    numerator: float,
    denominator: float,
    as_percent: bool = False,
) -> Optional[float]:
    """Compute a financial ratio safely."""
    if denominator == 0:
        return None
    result = numerator / denominator
    return result * 100 if as_percent else result
