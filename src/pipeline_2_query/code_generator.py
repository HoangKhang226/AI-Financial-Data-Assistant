"""
ViFinQA Code Generator
──────────────────────────────────────────────────────────────
Qwen2.5-Coder-14B-Instruct → sinh mã Python Pandas.
"""

import re
from typing import Optional
from src.common.logger import get_logger

logger = get_logger(__name__)


def generate_code(
    prompt: str,
    llm_client=None,
    system_prompt: str = "",
) -> str:
    """Generate Pandas code from a prompt using Code LLM.

    Returns extracted Python code string.
    """
    if llm_client is None:
        logger.warning("No LLM client provided, returning empty code")
        return ""

    raw_output = llm_client.generate(prompt, system_prompt)
    return extract_python_code(raw_output)


def extract_python_code(text: str) -> str:
    """Extract Python code block from LLM output.

    Handles ```python...``` blocks and bare code.
    """
    # Try to extract from markdown code blocks
    pattern = r"```(?:python|py)?\s*\n(.*?)\n?```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()

    # If no markdown block is found, see if it just returned raw code
    if "import pandas" in text or "df[" in text or "result =" in text:
        return text.strip()
        
    logger.error("Failed to extract Python code from LLM output")
    return ""
