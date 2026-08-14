"""
ViFinQA Entity Extractor
──────────────────────────────────────────────────────────────
Extracts structured metadata from Vietnamese financial questions:
  - Ticker(s), Year(s), Report Type, Target Unit, Complexity

Refactored from: AIGure_S2/question_parser.py
"""

import re
import json
import pandas as pd
from typing import List, Optional, Tuple
from src.common.logger import get_logger

logger = get_logger(__name__)

# ── Report Type Keywords ──
CONSOLIDATED_KW = ["hợp nhất", "consolidated", "tập đoàn"]
SEPARATE_KW = ["công ty mẹ", "riêng", "separate", "mother company"]

# ── Unit Conversion Map ──
UNIT_MAP = {
    "đồng": 1, "nghìn đồng": 1_000, "triệu đồng": 1_000_000,
    "tỷ đồng": 1_000_000_000, "trăm tỷ đồng": 100_000_000_000,
    "nghìn tỷ đồng": 1_000_000_000_000,
    "phần trăm": "percent", "%": "percent",
    "lần": "ratio", "điểm phần trăm": "pct_point",
    "doanh nghiệp": "count", "công ty": "count", "năm": "year",
}

# ── Alias Map: Nicknames / Brands → Ticker ──
ALIAS_MAP = {
    "vietcombank": "VCB", "ngoại thương": "VCB",
    "quân đội": "MBB", "mb": "MBB", "mbbank": "MBB",
    "việt nam thịnh vượng": "VPB", "vpbank": "VPB",
    "công thương": "CTG", "vietinbank": "CTG",
    "đầu tư và phát triển": "BID", "bidv": "BID",
    "á châu": "ACB",
    "phát triển thành phố hồ chí minh": "HDB", "hdbank": "HDB",
    "sài gòn - hà nội": "SHB", "sài gòn hà nội": "SHB",
    "đông nam á": "SSB", "seabank": "SSB",
    "quốc tế việt nam": "VIB", "hàng hải": "MSB",
    "phương đông": "OCB", "nam á": "NAB", "quốc dân": "NVB",
    "sài gòn tài lộc": "STB", "sacombank": "STB",
    "xuất nhập khẩu": "EIB", "eximbank": "EIB",
    "an bình": "ABB", "abbank": "ABB",
    "việt á": "VAB", "bắc á": "BAB", "kiên long": "KLB",
    "hòa phát": "HPG", "vingroup": "VIC",
    "cảng hàng không": "ACV", "ssi": "SSI",
    "khí việt nam": "GAS", "pvgas": "GAS",
    "nova": "NVL", "novaland": "NVL",
    "sữa việt nam": "VNM", "vinamilk": "VNM",
    "fpt": "FPT", "thế giới di động": "MWG", "masan": "MSN",
    "xăng dầu": "PLX", "petrolimex": "PLX",
    "sabeco": "SAB", "đất xanh group": "DXG",
    "hoàng anh gia lai": "HAG",
    "hàng không vietjet": "VJC", "vietjet": "VJC",
    "bảo việt": "BVH", "đạm phú mỹ": "DPM",
    "đạm cà mau": "DCM", "phú nhuận jewelry": "PNJ",
    "nam long": "NLG", "kinh bắc": "KBC",
    "hoa sen": "HSG", "viglacera": "VGC",
    "nhựa an phát xanh": "AAA", "an phát": "AAA",
    "đường quảng ngãi": "QNS", "lọc hóa dầu": "BSR",
    "hoà phát": "HPG", # typo variation
    "phân bón dầu khí cà mau": "DCM",
    "phân bón và hóa chất dầu khí": "DPM",
    "công nghiệp cao su việt nam": "GVR",
    "hpx": "HPX", "kbc": "KBC", "nvl": "NVL", "vic": "VIC", "vpi": "VPI", "vre": "VRE",
}


class EntityExtractor:
    """Extracts structured entities from Vietnamese financial questions."""

    def __init__(self, code_stock_path: str = ""):
        self.ticker_set: set = set()
        self.name_to_ticker: dict = {}

        if code_stock_path:
            self._load_code_stock(code_stock_path)

        # Merge alias map
        for alias, ticker in ALIAS_MAP.items():
            self.name_to_ticker[alias.lower()] = ticker

        # Pre-compile patterns
        self._year_pat = re.compile(r"\b(20[0-2]\d)\b")
        self._year_range_pat = re.compile(
            r"giai đoạn\s+(20[0-2]\d)\s*[-–]\s*(20[0-2]\d)"
        )
        self._ticker_pat = re.compile(r"\b([A-Z][A-Z0-9]{1,3})\b")
        self._ticker_parens = re.compile(r"\(([A-Z][A-Z0-9]{1,3})\)")

    def _load_code_stock(self, path: str) -> None:
        import os
        if not os.path.exists(path):
            logger.warning(f"code_stock file not found: {path}")
            return
        df = pd.read_csv(path)
        self.ticker_set = set(df["Mã CK"].tolist())
        for _, row in df.iterrows():
            name_lower = row["Tên công ty"].lower()
            ticker = row["Mã CK"]
            self.name_to_ticker[name_lower] = ticker
            for prefix in [
                "ctcp ", "ngân hàng tmcp ", "tổng công ty cổ phần ",
                "tổng công ty ", "tập đoàn ", "công ty tài chính ",
            ]:
                if name_lower.startswith(prefix):
                    stripped = name_lower[len(prefix):].strip()
                    if stripped:
                        self.name_to_ticker[stripped] = ticker

    def extract(self, qid: int, question: str) -> dict:
        """Extract all entities from a question.

        Returns dict with: tickers, years, report_type, target_unit,
        unit_type, complexity, year_range.
        """
        tickers = self._extract_tickers(question)
        years, year_range = self._extract_years(question)
        report_type = self._extract_report_type(question)
        target_unit, unit_type = self._extract_unit(question)
        complexity = self._classify_complexity(
            question, tickers, years, year_range, unit_type
        )

        return {
            "id": qid,
            "question": question,
            "tickers": tickers,
            "years": years,
            "report_type": report_type,
            "target_unit": target_unit,
            "unit_type": unit_type,
            "complexity": complexity,
            "year_range": year_range,
        }

    def _extract_tickers(self, question: str) -> List[str]:
        tickers, seen = [], set()
        # If no code_stock loaded, accept any uppercase 2-4 letter code
        no_stock = not self.ticker_set

        # Strategy 1: Ticker in parentheses
        for m in self._ticker_parens.finditer(question):
            t = m.group(1)
            if (no_stock or t in self.ticker_set) and t not in seen:
                tickers.append(t); seen.add(t)

        # Strategy 2: Standalone uppercase tickers
        ignore_words = {"TMCP", "CTCP", "TNHH", "BCTC", "VNĐ", "VND", "USD", "Q1", "Q2", "Q3", "Q4"}
        for m in self._ticker_pat.finditer(question):
            t = m.group(1)
            if t in ignore_words:
                continue
            if (no_stock or t in self.ticker_set) and t not in seen:
                tickers.append(t); seen.add(t)

        # Strategy 3: Company name/alias match
        if not tickers:
            q_lower = question.lower()
            for alias in sorted(self.name_to_ticker, key=len, reverse=True):
                if alias in q_lower:
                    t = self.name_to_ticker[alias]
                    if t not in seen:
                        tickers.append(t); seen.add(t)
                    break

        # Strategy 4: Group ticker lists
        group_pats = [
            re.compile(r"\(([A-Z]{2,4}(?:\s*,\s*[A-Z]{2,4})+)\)"),
            re.compile(
                r"(?:nhóm|các công ty|các doanh nghiệp)\s+"
                r"([A-Z][A-Z0-9]{1,3}(?:\s*[,và\s]+\s*[A-Z][A-Z0-9]{1,3})+)"
            ),
        ]
        for pat in group_pats:
            m = pat.search(question)
            if m:
                for t in re.findall(r"[A-Z][A-Z0-9]{1,3}", m.group(1)):
                    if (no_stock or t in self.ticker_set) and t not in seen:
                        tickers.append(t); seen.add(t)

        return tickers

    def _extract_years(self, question: str) -> Tuple[List[int], Optional[Tuple[int, int]]]:
        years, year_range = [], None

        rm = self._year_range_pat.search(question)
        if rm:
            s, e = int(rm.group(1)), int(rm.group(2))
            year_range = (s, e)
            years = list(range(s, e + 1))
        else:
            for m in self._year_pat.finditer(question):
                y = int(m.group(1))
                if y not in years:
                    years.append(y)
            rm2 = re.search(r"(20[0-2]\d)\s*[-–]\s*(20[0-2]\d)", question)
            if rm2:
                s, e = int(rm2.group(1)), int(rm2.group(2))
                year_range = (s, e)
                for y in range(s, e + 1):
                    if y not in years:
                        years.append(y)

        years.sort()
        return years, year_range

    def _extract_report_type(self, question: str) -> str:
        q = question.lower()
        for kw in SEPARATE_KW:
            if kw in q:
                return "separate"
        for kw in CONSOLIDATED_KW:
            if kw in q:
                return "consolidated"
        return ""

    def _extract_unit(self, question: str) -> Tuple[str, str]:
        q = question.lower()
        for unit_str in sorted(UNIT_MAP, key=len, reverse=True):
            pat = rf"(?:là|bằng|bao nhiêu)\s+(?:bao nhiêu\s+)?{re.escape(unit_str)}"
            if re.search(pat, q):
                val = UNIT_MAP[unit_str]
                if isinstance(val, int):
                    return unit_str, "absolute"
                return unit_str, str(val)

        end_m = re.search(
            r"bao nhiêu\s+(triệu đồng|tỷ đồng|đồng|phần trăm|lần"
            r"|doanh nghiệp|công ty)\s*\??\s*$", q
        )
        if end_m:
            u = end_m.group(1)
            val = UNIT_MAP.get(u, 1)
            return u, ("absolute" if isinstance(val, int) else str(val))

        return "", "unknown"

    def _classify_complexity(
        self, question: str, tickers: list, years: list,
        year_range, unit_type: str,
    ) -> str:
        q = question.lower()
        if len(tickers) > 2:
            if any(kw in q for kw in [
                "trung vị", "bình quân", "tỷ trọng",
                "bao nhiêu doanh nghiệp", "bao nhiêu công ty",
            ]):
                return "multi_company_aggregate"
            return "multi_company"
        if year_range or len(years) >= 3:
            return "time_series"
        if len(years) == 2 and any(
            kw in q for kw in ["tăng trưởng", "thay đổi", "chênh lệch", "so với"]
        ):
            return "comparative"
        if len(tickers) == 1 and len(years) == 1:
            return "simple"
        if any(kw in q for kw in [
            "cfo dương", "cfo âm", "duy trì", "đồng thời",
            "thấp hơn", "cao hơn", "trên mức",
        ]):
            return "conditional"
        return "complex"


def parse_questions_file(
    questions_path: str, code_stock_path: str = "",
) -> List[dict]:
    """Parse all questions from a JSONL file."""
    extractor = EntityExtractor(code_stock_path)
    results = []
    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            parsed = extractor.extract(obj["id"], obj["question"])
            results.append(parsed)
    return results
