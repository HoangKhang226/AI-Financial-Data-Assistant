"""
ViFinQA Prompt Builder
──────────────────────────────────────────────────────────────
Builds XML-structured prompts for Code LLM with:
  - DataFrame schema, sample rows, unit info, question
"""

import pandas as pd
from typing import List, Optional


def build_code_prompt(
    question: str,
    df: pd.DataFrame,
    unit: str = "đồng",
    target_unit: str = "",
    max_sample_rows: int = 5,
) -> str:
    """Build a prompt for Pandas code generation.

    Returns XML-structured prompt string.
    """
    # Schema info
    schema_lines = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        schema_lines.append(f"  <column name=\"{col}\" dtype=\"{dtype}\"/>")
    schema_xml = "\n".join(schema_lines)

    # Sample rows
    sample = df.head(max_sample_rows).to_string(index=False)

    prompt = f"""<task>
Generate Python Pandas code to answer the financial question.
Store the final numeric answer in a variable called `result`.
</task>

<dataframe_schema>
{schema_xml}
</dataframe_schema>

<sample_data>
{sample}
</sample_data>

<unit_info>
  <source_unit>{unit}</source_unit>
  <target_unit>{target_unit or unit}</target_unit>
</unit_info>

<question>{question}</question>

<rules>
- The dataframe is ALREADY loaded in the variable `df`. Do NOT use pd.read_csv.
- ONLY use pandas operations on `df`. Store the final numeric answer in `result`.
- ALWAYS use `.iloc` for column access if you are not sure of the exact column name. The first column is usually the metric name (e.g., `df.iloc[:, 0]`), and subsequent columns are values for specific years.
- Do NOT search for exact full strings from the question (e.g. "Tổng tài sản AAA năm 2017"). Instead, use `str.contains()` on the first column to find matching rows (e.g. `df.iloc[:, 0].astype(str).str.contains('Tổng tài sản', case=False, na=False)`).
- Handle Vietnamese number format (dots = thousands, comma = decimal) and negative values in parentheses: `(1.234)` means `-1234`.
- If the mask matches nothing (i.e. `not row_mask.any()`), do NOT access `.values[0]`. Assign `result = None`.
- Do NOT use print().
</rules>

<example_code>
```python
import pandas as pd
import numpy as np

# Find the row containing the financial metric using keywords from the question
# Example: If the question asks for "Lợi nhuận", search for "Lợi nhuận"
row_mask = df.iloc[:, 0].astype(str).str.contains('YOUR_METRIC_KEYWORD', case=False, na=False)
if not row_mask.any():
    row_mask = df.iloc[:, 0].astype(str).str.contains('ALTERNATIVE_KEYWORD', case=False, na=False)

if row_mask.any():
    # Target the column for the requested year
    year_cols = [col for col in df.columns if 'YOUR_YEAR' in str(col)]
    target_col = year_cols[0] if year_cols else df.columns[1]
    
    raw_value = df.loc[row_mask, target_col].values[0]
    
    if pd.isna(raw_value):
        result = None
    else:
        val_str = str(raw_value).replace('.', '').strip()
        if '(' in val_str and ')' in val_str:
            val_str = '-' + val_str.replace('(', '').replace(')', '')
        result = float(val_str.replace(',', '.'))
else:
    result = None
```
</example_code>"""

    return prompt


def build_correction_prompt(
    original_code: str,
    error_traceback: str,
    df_schema: str,
) -> str:
    """Build a prompt for self-correction of failed code."""
    return f"""<task>Fix the Python code that produced an error.</task>

<previous_code>
{original_code}
</previous_code>

<error>
{error_traceback}
</error>

<schema>
{df_schema}
</schema>

<rules>
- Fix the error and return only corrected Python code
- Keep using the `df` variable and store result in `result`
</rules>"""
