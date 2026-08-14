"""
ViFinQA Sandbox Executor
──────────────────────────────────────────────────────────────
Executes generated Python code in an isolated subprocess
with timeout and stderr capture.
"""

import subprocess
import sys
import tempfile
import os
from typing import Any, Optional, Tuple
from src.common.logger import get_logger

logger = get_logger(__name__)


def execute_code(
    code: str,
    df_csv_path: str,
    timeout: int = 5,
) -> Tuple[bool, Any, str]:
    """Execute Pandas code in a restricted subprocess.

    Args:
        code: Python code to execute (must store result in `result`).
        df_csv_path: Path to CSV file to load as `df`.
        timeout: Max execution time in seconds.

    Returns:
        (success, result_value, error_message)
    """
    abs_csv_path = os.path.abspath(df_csv_path)
    # Build the wrapper script
    wrapper = f"""
import pandas as pd
import sys
import json

df = pd.read_csv(r"{abs_csv_path}", encoding="utf-8-sig")

{code}

# Output result
if 'result' in dir():
    print(json.dumps({{"result": result if not isinstance(result, pd.DataFrame) else result.to_dict()}}))
else:
    print(json.dumps({{"error": "No 'result' variable defined"}}))
"""

    try:
        proc = subprocess.run(
            [sys.executable, "-c", wrapper],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(abs_csv_path) or ".",
        )

        if proc.returncode != 0:
            return False, None, proc.stderr.strip()

        import json
        try:
            output = json.loads(proc.stdout.strip())
            if "error" in output:
                return False, None, output["error"]
            return True, output["result"], ""
        except (json.JSONDecodeError, KeyError):
            return False, None, f"Invalid output: {proc.stdout[:200]}"

    except subprocess.TimeoutExpired:
        return False, None, f"Execution timeout ({timeout}s)"
    except Exception as e:
        return False, None, str(e)
