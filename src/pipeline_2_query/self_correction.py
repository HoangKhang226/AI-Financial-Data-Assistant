"""
ViFinQA Self-Correction Loop
──────────────────────────────────────────────────────────────
Sends traceback + failed code back to LLM for correction.
Maximum 3 retries.
"""

from typing import Tuple, Any, Optional
from src.common.logger import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 3


def self_correct(
    original_code: str,
    error_msg: str,
    df_csv_path: str,
    llm_client=None,
    prompt_builder_fn=None,
    executor_fn=None,
    df_schema: str = "",
) -> Tuple[bool, Any, str, int]:
    """Run self-correction loop on failed code.

    Returns (success, result, final_code, retries_used).
    """
    if llm_client is None or prompt_builder_fn is None or executor_fn is None:
        return False, None, original_code, 0

    current_code = original_code
    current_error = error_msg

    for retry in range(1, MAX_RETRIES + 1):
        logger.info(f"Self-correction retry {retry}/{MAX_RETRIES}")

        correction_prompt = prompt_builder_fn(
            current_code, current_error, df_schema
        )

        from src.pipeline_2_query.code_generator import generate_code
        new_code = generate_code(correction_prompt, llm_client)

        if not new_code:
            continue

        success, result, error = executor_fn(new_code, df_csv_path)
        if success:
            return True, result, new_code, retry

        current_code = new_code
        current_error = error

    return False, None, current_code, MAX_RETRIES
