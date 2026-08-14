"""
ViFinQA Pipeline 2 Orchestrator
──────────────────────────────────────────────────────────────
End-to-end query pipeline using LangGraph state machine:
  Question -> Parse -> Search -> CodeGen -> Execute
    -> (Reflect -> ReGen -> Execute) x 2
    -> Fallback RAG -> Format

Supports checkpoint-based resume across Colab sessions.
"""

import json
import os
import time
from typing import Dict
from src.common.logger import get_logger
from src.pipeline_2_query.submission_formatter import (
    build_submission, save_submission,
)

logger = get_logger(__name__)


def run_query(
    questions_path: str,
    code_stock_path: str = "",
    metadata_dir: str = "data/metadata",
    csv_warehouse_dir: str = "data/csv_warehouse",
    output_path: str = "output/submission.json",
    checkpoint_path: str = "output/checkpoint_results.jsonl",
    index_dir: str = "data/index",
    use_llm: bool = False,
    verbose: bool = False,
) -> Dict:
    """Run Pipeline 2 on all questions with LangGraph + checkpointing.

    Args:
        questions_path: Path to questions.jsonl.
        code_stock_path: Path to code_stock.csv.
        metadata_dir: Path to metadata JSON directory.
        csv_warehouse_dir: Path to CSV warehouse directory.
        output_path: Where to save submission.json.
        checkpoint_path: Path to append-only JSONL checkpoint file.
        use_llm: Whether to load and use LLM for code generation.
        verbose: Print progress.

    Returns:
        Submission dict.
    """
    # ── Initialize Components ──
    from src.pipeline_2_query.entity_extractor import EntityExtractor
    from src.pipeline_2_query.metadata_filter import MetadataFilter

    extractor = EntityExtractor(code_stock_path)
    meta_filter = MetadataFilter(metadata_dir)

    llm_client = None
    searcher = None
    embedding_client = None
    reranker_client = None

    if use_llm:
        try:
            from src.common.llm_client import LLMClient
            from src.common.embedding_client import EmbeddingClient
            from src.common.reranker_client import RerankerClient
            from src.pipeline_2_query.hybrid_searcher import HybridSearcher

            llm_client = LLMClient()
            embedding_client = EmbeddingClient(device="cpu")
            reranker_client = RerankerClient(device="cpu")
            searcher = HybridSearcher(
                dense_dir=os.path.join(index_dir, "dense_vectors"),
                bm25_dir=os.path.join(index_dir, "bm25_index"),
            )
            searcher.load()
            logger.info("LLM pipeline initialized (vLLM + BGE-M3 + BM25 + Reranker)")
        except Exception as e:
            logger.error(f"Failed to initialize LLM pipeline: {e}")
            logger.info("Falling back to deterministic extraction mode")

    # ── Build LangGraph ──
    from src.pipeline_2_query.graph import build_query_graph

    graph = build_query_graph(
        extractor=extractor,
        meta_filter=meta_filter,
        searcher=searcher,
        embedding_client=embedding_client,
        reranker_client=reranker_client,
        llm_client=llm_client,
        csv_warehouse_dir=csv_warehouse_dir,
    )
    logger.info("LangGraph query pipeline compiled")

    # ── Load Checkpoints ──
    completed_ids = set()
    results = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        res = json.loads(line)
                        completed_ids.add(res["question_id"])
                        results.append(res)
                    except json.JSONDecodeError:
                        pass
        logger.info(f"Loaded {len(completed_ids)} completed questions from checkpoint.")

    # ── Load Questions ──
    questions = []
    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                if q["id"] not in completed_ids:
                    questions.append(q)

    stats = {
        "total": len(completed_ids),
        "success": sum(1 for r in results if r.get("success")),
        "failed": sum(1 for r in results if not r.get("success")),
    }

    if verbose:
        if questions:
            print(f"Resuming: {len(questions)} remaining questions (skipped {len(completed_ids)} from checkpoint)")
        else:
            print(f"All {len(completed_ids)} questions already answered in checkpoint!")

    # ── Main Processing Loop ──
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    with open(checkpoint_path, "a", encoding="utf-8") as chk_file:
        for i, q in enumerate(questions):
            start = time.time()
            qid = q["id"]
            question = q["question"]

            if verbose:
                print(f"[{i+1}/{len(questions)}] Q{qid}: {question[:80]}...")

            # Run through LangGraph
            try:
                initial_state = {
                    "question_id": str(qid),
                    "question": question,
                    "tickers": [],
                    "years": [],
                    "report_type": "",
                    "target_unit": "",
                    "unit_type": "unknown",
                    "candidate_doc_ids": [],
                    "top_csv_path": "",
                    "df_schema": "",
                    "generated_code": "",
                    "exec_success": False,
                    "exec_result": None,
                    "exec_error": "",
                    "retry_count": 0,
                    "error_analysis": "",
                    "final_answer": "0",
                }

                final_state = graph.invoke(initial_state)
                answer_str = final_state.get("final_answer", "0")
                success = answer_str != "0" and answer_str != ""

            except Exception as e:
                logger.error(f"LangGraph error on Q{qid}: {e}")
                answer_str = "0"
                success = False

            elapsed = (time.time() - start) * 1000

            # Fetch metadata for the top CSV to build submission details
            relevant_docs = []
            relevant_tables = []
            evidence = []
            pandas_query = final_state.get("generated_code", "")

            top_csv = final_state.get("top_csv_path", "")
            if top_csv and os.path.exists(top_csv):
                # We need to construct evidence array
                evidence.append({
                    "variable": "df",
                    "csv_path": f"data/{os.path.basename(top_csv)}"
                })
                
                # Fetch metadata to get source_file and line_number
                meta_path = os.path.join(
                    "data/metadata", 
                    os.path.basename(top_csv).replace(".csv", ".json")
                )
                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    source_file = meta.get("source_file", "")
                    if source_file:
                        doc_id = os.path.basename(source_file)
                        if doc_id.endswith(".txt"):
                            doc_id = doc_id[:-4]
                        
                        relevant_docs.append(doc_id)
                        line_num = meta.get("line_number", 0)
                        relevant_tables.append(f"{doc_id}|{line_num}")

            result = {
                "id": int(qid) if str(qid).isdigit() else qid,
                "question_id": qid,
                "question": question,
                "answer": answer_str,
                "formatted_answer": answer_str,
                "success": success,
                "time_ms": elapsed,
                "retries": final_state.get("retry_count", 0) if 'final_state' in dir() else 0,
                "relevant_docs": relevant_docs,
                "relevant_tables": relevant_tables,
                "evidence": evidence,
                "pandas_query": pandas_query,
            }
            results.append(result)

            # Checkpoint immediately
            chk_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            chk_file.flush()

            if success:
                stats["success"] += 1
            else:
                stats["failed"] += 1
            stats["total"] += 1

            if verbose:
                status = "OK" if success else "FAIL"
                retries = result.get("retries", 0)
                retry_info = f" (retries={retries})" if retries > 0 else ""
                print(f"  [{status}] {answer_str} | {elapsed:.0f}ms{retry_info}")

    # ── Build Final Submission ──
    submission = build_submission(results)
    save_submission(submission, output_path)

    logger.info(
        f"Pipeline 2 complete: {stats['success']}/{stats['total']} answered"
    )
    return submission


if __name__ == "__main__":
    import argparse
    import sys

    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    def abs_path(p):
        return os.path.join(root_dir, p) if p and not os.path.isabs(p) else p

    parser = argparse.ArgumentParser(description="Run Pipeline 2 Query (LangGraph)")
    parser.add_argument("--questions", type=str, default=abs_path("data/test_questions.jsonl"))
    parser.add_argument("--code-stock", type=str, default=abs_path("data/ViFinQA/code_stock.csv"))
    parser.add_argument("--metadata", type=str, default=abs_path("data/metadata"))
    parser.add_argument("--csv-warehouse", type=str, default=abs_path("data/csv_warehouse"))
    parser.add_argument("--index", type=str, default=abs_path("data/index"))
    parser.add_argument("--output", type=str, default=abs_path("output/submission.json"))
    parser.add_argument("--checkpoint", type=str, default=abs_path("output/checkpoint_results.jsonl"))
    parser.add_argument("--use-llm", action="store_true", help="Enable LLM code generation")
    parser.add_argument("--verbose", action="store_true", default=True)

    args = parser.parse_args()

    print("=" * 60)
    print("STARTING PIPELINE 2: QUERY (LangGraph)")
    print("=" * 60)

    run_query(
        questions_path=args.questions,
        code_stock_path=args.code_stock,
        metadata_dir=args.metadata,
        csv_warehouse_dir=args.csv_warehouse,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        index_dir=args.index,
        use_llm=args.use_llm,
        verbose=args.verbose,
    )
