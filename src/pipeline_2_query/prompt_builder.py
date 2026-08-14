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
- Use only pandas operations on the variable `df`
- Handle Vietnamese number format (dots = thousands, comma = decimal)
- Handle negative values in parentheses: (1.234) means -1234
- Store final answer in `result` variable
- Do NOT use print()
</rules>"""

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
