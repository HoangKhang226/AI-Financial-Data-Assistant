"""
ViFinQA Metadata Filter
──────────────────────────────────────────────────────────────
Hard-filters JSON metadata by ticker + year + report_type
before vector/sparse search.

Refactored from: AIGure_S2/report_retriever.py
"""

import os
import json
import glob
from typing import List, Optional
from src.common.logger import get_logger

logger = get_logger(__name__)


class MetadataFilter:
    """Filters metadata JSON store by structured criteria."""

    def __init__(self, metadata_dir: str = "data/metadata"):
        self.metadata_dir = metadata_dir
        self._index: List[dict] = []
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        pattern = os.path.join(self.metadata_dir, "*.json")
        for fp in glob.glob(pattern):
            with open(fp, "r", encoding="utf-8") as f:
                self._index.append(json.load(f))
        self._loaded = True
        logger.info(f"Loaded {len(self._index)} metadata entries")

    def filter(
        self,
        ticker: str = "",
        year: Optional[int] = None,
        report_type: str = "",
        category: str = "",
    ) -> List[dict]:
        """Filter metadata by criteria. Empty/None = no filter."""
        self.load()
        results = self._index

        if ticker:
            results = [m for m in results if m["ticker"] == ticker]
        if year is not None:
            results = [m for m in results if m["year"] == year]
        if report_type:
            results = [m for m in results if m["report_type"] == report_type]
        if category:
            results = [m for m in results if m["category"] == category]

        return results

    def get_doc_ids(
        self, ticker: str = "", year: Optional[int] = None,
        report_type: str = "",
    ) -> List[str]:
        """Get filtered doc_ids for downstream search scoping."""
        return [m["doc_id"] for m in self.filter(ticker, year, report_type)]

    def get_available_years(self, ticker: str) -> List[int]:
        self.load()
        years = set()
        for m in self._index:
            if m["ticker"] == ticker:
                years.add(m["year"])
        return sorted(years)

    def has_report(
        self, ticker: str, year: int, report_type: str = ""
    ) -> bool:
        return bool(self.filter(ticker, year, report_type))
