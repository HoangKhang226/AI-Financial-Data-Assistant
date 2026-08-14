"""
ViFinQA LangGraph Query Graph
──────────────────────────────────────────────────────────────
State machine for the full query pipeline with self-correction:

  Parse → Search → Rerank → CodeGen → Execute
    ↓ (error, retry < 2)
  Reflect → ReGen → Execute
    ↓ (error, retry >= 2)
  Fallback RAG → Format
"""

import os
import pandas as pd
from typing import TypedDict, Optional, List, Literal
from langgraph.graph import StateGraph, END
from src.common.logger import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 2


# ══════════════════════════════════════════════════════════════
# State Schema
# ══════════════════════════════════════════════════════════════

class QueryState(TypedDict):
    """State flowing through the LangGraph pipeline."""
    # ── Input ──
    question_id: str
    question: str

    # ── Entity extraction ──
    tickers: List[str]
    years: List[int]
    report_type: str
    target_unit: str
    unit_type: str

    # ── Search output ──
    candidate_doc_ids: List[str]
    top_csv_path: str
    df_schema: str

    # ── Code generation ──
    generated_code: str

    # ── Execution ──
    exec_success: bool
    exec_result: Optional[float]
    exec_error: str

    # ── Self-correction ──
    retry_count: int
    error_analysis: str

    # ── Final ──
    final_answer: str


# ══════════════════════════════════════════════════════════════
# Node Functions
# ══════════════════════════════════════════════════════════════

def parse_question(state: QueryState, **kwargs) -> dict:
    """Node 1: Extract entities from the question."""
    extractor = kwargs.get("extractor")
    if extractor is None:
        from src.pipeline_2_query.entity_extractor import EntityExtractor
        extractor = EntityExtractor()

    parsed = extractor.extract(state["question_id"], state["question"])
    return {
        "tickers": parsed["tickers"],
        "years": parsed["years"],
        "report_type": parsed["report_type"],
        "target_unit": parsed.get("target_unit", ""),
        "unit_type": parsed.get("unit_type", "unknown"),
    }


def hybrid_search(state: QueryState, **kwargs) -> dict:
    """Node 2+3: Metadata filter → Hybrid search → Rerank → Pick top CSV."""
    meta_filter = kwargs.get("meta_filter")
    searcher = kwargs.get("searcher")
    embedding_client = kwargs.get("embedding_client")
    reranker_client = kwargs.get("reranker_client")
    csv_warehouse_dir = kwargs.get("csv_warehouse_dir", "data/csv_warehouse")

    # Step A: Metadata filter
    candidate_doc_ids = []
    if meta_filter:
        for ticker in state["tickers"]:
            for year in state["years"]:
                doc_ids = meta_filter.get_doc_ids(
                    ticker, year, state["report_type"]
                )
                candidate_doc_ids.extend(doc_ids)

    if not candidate_doc_ids:
        logger.warning(f"Q{state['question_id']}: No metadata candidates")
        return {
            "candidate_doc_ids": [],
            "top_csv_path": "",
            "df_schema": "",
        }

    # Step B: Hybrid search (if searcher available)
    top_doc_id = candidate_doc_ids[0]  # default: first candidate
    if searcher:
        search_results = searcher.search(
            state["question"], top_k=10,
            scope_doc_ids=candidate_doc_ids,
            embedding_client=embedding_client,
        )
        if search_results:
            # Rerank if available
            if reranker_client:
                from src.pipeline_2_query.reranker import rerank_documents
                summaries_map = _load_summaries_map()
                doc_ids = [did for did, _ in search_results[:10]]
                doc_texts = [summaries_map.get(did, "") for did in doc_ids]
                reranked = rerank_documents(
                    state["question"], doc_ids, doc_texts,
                    reranker_client=reranker_client, top_k=3,
                )
                if reranked:
                    top_doc_id = reranked[0][0]
            else:
                top_doc_id = search_results[0][0]

    # Step C: Load CSV and build schema
    csv_path = os.path.join(csv_warehouse_dir, f"{top_doc_id}.csv")
    df_schema = ""
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            schema_lines = [f"Columns: {list(df.columns)}"]
            schema_lines.append(f"Shape: {df.shape}")
            schema_lines.append(f"Sample:\n{df.head(3).to_string(index=False)}")
            df_schema = "\n".join(schema_lines)
        except Exception:
            pass

    return {
        "candidate_doc_ids": candidate_doc_ids,
        "top_csv_path": csv_path if os.path.exists(csv_path) else "",
        "df_schema": df_schema,
    }


def generate_code(state: QueryState, **kwargs) -> dict:
    """Node 4: Generate Pandas code using LLM."""
    llm_client = kwargs.get("llm_client")
    if not llm_client or not state["top_csv_path"]:
        return {"generated_code": ""}

    from src.pipeline_2_query.prompt_builder import build_code_prompt

    df = pd.read_csv(state["top_csv_path"], encoding="utf-8-sig")
    prompt = build_code_prompt(
        state["question"], df,
        unit=state.get("target_unit", ""),
    )

    from src.pipeline_2_query.code_generator import generate_code as gen
    
    system_prompt = "You are an expert Python Pandas data analyst. Always output ONLY valid Python code inside ```python ``` blocks. Do not explain."
    
    code = gen(prompt, llm_client, system_prompt=system_prompt)
    return {"generated_code": code}


def execute_code(state: QueryState, **kwargs) -> dict:
    """Node 5: Execute generated code in sandbox."""
    if not state["generated_code"] or not state["top_csv_path"]:
        return {
            "exec_success": False,
            "exec_result": None,
            "exec_error": "No code or CSV path",
        }

    from src.pipeline_2_query.sandbox_executor import execute_code as exec_fn
    success, result, error = exec_fn(state["generated_code"], state["top_csv_path"])

    exec_result = None
    if success and result is not None:
        try:
            exec_result = float(result)
        except (ValueError, TypeError):
            success = False
            error = f"Cannot convert result to float: {result}"

    return {
        "exec_success": success,
        "exec_result": exec_result,
        "exec_error": error,
    }


def reflect_on_error(state: QueryState, **kwargs) -> dict:
    """Node 6: LLM analyzes WHY the code failed (root cause analysis)."""
    llm_client = kwargs.get("llm_client")
    if not llm_client:
        return {"error_analysis": state["exec_error"]}

    reflect_prompt = f"""<task>Phan tich loi trong doan code Python xu ly du lieu tai chinh.</task>

<failed_code>
{state["generated_code"]}
</failed_code>

<error>
{state["exec_error"]}
</error>

<dataframe_schema>
{state["df_schema"]}
</dataframe_schema>

<instructions>
1. Xac dinh nguyen nhan goc re cua loi (sai ten cot? sai kieu du lieu? logic sai?)
2. De xuat cach sua cu the (1-2 cau ngan gon)
3. Chu y: Du lieu tai chinh Viet Nam dung dau cham phan cach hang nghin, dau ngoac () la so am
</instructions>"""

    analysis = llm_client.generate(reflect_prompt)
    logger.info(f"Q{state['question_id']} Reflect: {analysis[:100]}...")

    return {"error_analysis": analysis}


def regenerate_code(state: QueryState, **kwargs) -> dict:
    """Node 7: Re-generate code with error analysis context."""
    llm_client = kwargs.get("llm_client")
    if not llm_client:
        return {"generated_code": "", "retry_count": state["retry_count"] + 1}

    regen_prompt = f"""<task>Sua lai code Python dua tren phan tich loi.</task>

<error_analysis>{state["error_analysis"]}</error_analysis>

<previous_code>
{state["generated_code"]}
</previous_code>

<traceback>
{state["exec_error"]}
</traceback>

<dataframe_schema>
{state["df_schema"]}
</dataframe_schema>

<question>{state["question"]}</question>

<rules>
- Ap dung cach sua duoc de xuat trong error_analysis
- Luu ket qua vao bien `result`
- Xu ly format so Viet Nam (dau cham = hang nghin, ngoac = so am)
- Chi tra ve code Python, khong giai thich
</rules>"""

    from src.pipeline_2_query.code_generator import generate_code as gen
    new_code = gen(regen_prompt, llm_client)

    return {
        "generated_code": new_code,
        "retry_count": state["retry_count"] + 1,
    }


def fallback_rag(state: QueryState, **kwargs) -> dict:
    """Node 8: Deterministic regex-based extraction as last resort."""
    csv_warehouse_dir = kwargs.get("csv_warehouse_dir", "data/csv_warehouse")

    from src.pipeline_2_query.fallback_rag import extract_from_csv

    question_lower = state["question"].lower()
    year_hint = state["years"][0] if state["years"] else None

    # Try each candidate CSV
    for doc_id in state["candidate_doc_ids"]:
        csv_path = os.path.join(csv_warehouse_dir, f"{doc_id}.csv")
        if not os.path.exists(csv_path):
            continue

        for item_name in [
            "doanh thu thuần", "lợi nhuận gộp", "lợi nhuận sau thuế",
            "lợi nhuận trước thuế", "tổng tài sản", "vốn chủ sở hữu",
            "nợ phải trả", "tiền và các khoản tương đương tiền",
            "hàng tồn kho", "chi phí tài chính", "giá vốn hàng bán",
            "thu nhập lãi thuần", "cho vay khách hàng",
        ]:
            if item_name in question_lower:
                value = extract_from_csv(csv_path, item_name, year_hint)
                if value is not None:
                    return {"final_answer": str(value)}

    return {"final_answer": "0"}


def format_answer(state: QueryState, **kwargs) -> dict:
    """Node 9: Format the final answer."""
    from src.pipeline_2_query.submission_formatter import format_answer as fmt

    if state.get("exec_success") and state.get("exec_result") is not None:
        answer = fmt(state["exec_result"], state.get("unit_type", "unknown"))
        return {"final_answer": answer}

    # Already set by fallback
    if state.get("final_answer") and state["final_answer"] != "":
        return {}

    return {"final_answer": "0"}


# ══════════════════════════════════════════════════════════════
# Routing Logic
# ══════════════════════════════════════════════════════════════

def route_after_execute(state: QueryState) -> Literal["format_answer", "reflect_on_error", "fallback_rag"]:
    """Conditional edge after code execution."""
    if state.get("exec_success"):
        return "format_answer"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        logger.info(f"Q{state['question_id']}: Max retries ({MAX_RETRIES}) reached, falling back")
        return "fallback_rag"
    logger.info(f"Q{state['question_id']}: Retry {state.get('retry_count', 0)+1}/{MAX_RETRIES}")
    return "reflect_on_error"


def route_after_search(state: QueryState) -> Literal["generate_code", "fallback_rag"]:
    """Skip LLM if no CSV found."""
    if state.get("top_csv_path"):
        return "generate_code"
    return "fallback_rag"


# ══════════════════════════════════════════════════════════════
# Graph Builder
# ══════════════════════════════════════════════════════════════

def build_query_graph(
    extractor=None,
    meta_filter=None,
    searcher=None,
    embedding_client=None,
    reranker_client=None,
    llm_client=None,
    csv_warehouse_dir: str = "data/csv_warehouse",
):
    """Build and compile the LangGraph query pipeline.

    All clients are injected via closures into node functions.
    """
    # Bind kwargs to node functions via closures
    deps = dict(
        extractor=extractor,
        meta_filter=meta_filter,
        searcher=searcher,
        embedding_client=embedding_client,
        reranker_client=reranker_client,
        llm_client=llm_client,
        csv_warehouse_dir=csv_warehouse_dir,
    )

    def _parse(state):
        return parse_question(state, **deps)

    def _search(state):
        return hybrid_search(state, **deps)

    def _codegen(state):
        return generate_code(state, **deps)

    def _execute(state):
        return execute_code(state, **deps)

    def _reflect(state):
        return reflect_on_error(state, **deps)

    def _regen(state):
        return regenerate_code(state, **deps)

    def _fallback(state):
        return fallback_rag(state, **deps)

    def _format(state):
        return format_answer(state, **deps)

    # ── Build Graph ──
    graph = StateGraph(QueryState)

    graph.add_node("parse_question", _parse)
    graph.add_node("hybrid_search", _search)
    graph.add_node("generate_code", _codegen)
    graph.add_node("execute_code", _execute)
    graph.add_node("reflect_on_error", _reflect)
    graph.add_node("regenerate_code", _regen)
    graph.add_node("fallback_rag", _fallback)
    graph.add_node("format_answer", _format)

    # ── Edges ──
    graph.set_entry_point("parse_question")
    graph.add_edge("parse_question", "hybrid_search")

    # After search: go to codegen if CSV found, else fallback
    graph.add_conditional_edges("hybrid_search", route_after_search)

    graph.add_edge("generate_code", "execute_code")

    # After execute: success → format, error → reflect or fallback
    graph.add_conditional_edges("execute_code", route_after_execute)

    graph.add_edge("reflect_on_error", "regenerate_code")
    graph.add_edge("regenerate_code", "execute_code")

    graph.add_edge("fallback_rag", "format_answer")
    graph.add_edge("format_answer", END)

    return graph.compile()


# ══════════════════════════════════════════════════════════════
# Helper
# ══════════════════════════════════════════════════════════════

def _load_summaries_map(
    summaries_path: str = "data/index/summaries.jsonl",
) -> dict:
    """Load doc_id → summary map for reranking."""
    import json
    summaries = {}
    if os.path.exists(summaries_path):
        with open(summaries_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    summaries[obj["doc_id"]] = obj.get("summary", "")
    return summaries
