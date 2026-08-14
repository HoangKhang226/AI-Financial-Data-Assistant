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
    pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()

    # If no code block, try to find lines that look like Python
    lines = text.strip().split("\n")
    code_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") or stripped.startswith("import"):
            code_lines.append(line)

    return "\n".join(code_lines).strip() if code_lines else text.strip()
