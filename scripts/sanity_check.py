import os
import sys
import argparse
import hashlib
from typing import List, Optional

from pathlib import Path
from dotenv import load_dotenv
import httpx
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Ensure project root is on sys.path so 'src' package can be imported
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load .env early
load_dotenv()

from src.vectorstore import VectorStoreManager


def _chunk_uid(doc: Document) -> Optional[str]:
    """Helper: return metadata['chunk_uid'] if present, else None."""
    try:
        meta = getattr(doc, "metadata", {}) or {}
        cid = meta.get("chunk_uid")
        return str(cid) if cid else None
    except Exception:
        return None


def _fallback_id(doc: Document) -> str:
    """
    Deterministic fallback identity mirroring VectorStoreManager._fallback_id():
    sha1("source|page|head200").
    """
    try:
        meta = getattr(doc, "metadata", {}) or {}
    except Exception:
        meta = {}
    src = meta.get("source", "")
    pg = str(meta.get("page", ""))
    sample = (getattr(doc, "page_content", "") or "")[:200]
    raw = f"{src}|{pg}|{sample}"
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()


def derive_identity(doc: Document) -> str:
    """
    Prefer metadata['chunk_uid']; else fallback hash.
    Mirrors VectorStoreManager._doc_identity().
    """
    cid = _chunk_uid(doc)
    return cid if cid else _fallback_id(doc)


def parse_cli_args() -> argparse.Namespace:
    """
    CLI for retrieval sanity checks.
    --mode {semantic,keyword,hybrid,all}
    --k int
    --dense-weight float
    --sparse-weight float
    --rrf-k int
    --queries str (comma-separated)

    Multi-Query Fusion options:
      --multi-query (flag): enable multi-query expansion/fusion
      --num-query-variants (int): number of expansion variants (>= 2 recommended when enabled)
      --expansion-method {heuristic,llm}
      --multi-query-rrf-k (int, optional): overrides RRF_K for multi-query fusion

    Examples:
      python scripts/sanity_check.py --mode hybrid --multi-query --num-query-variants 6
      python scripts/sanity_check.py --mode semantic --multi-query --expansion-method heuristic --k 5
      python scripts/sanity_check.py --mode keyword --multi-query --num-query-variants 8 --multi-query-rrf-k 80
    """
    # .env already loaded above; still safe to call again
    load_dotenv()
    parser = argparse.ArgumentParser(description="Sanity checks for retrieval modes (semantic | keyword | hybrid).")
    parser.add_argument(
        "--mode",
        choices=["semantic", "keyword", "hybrid", "all"],
        default=os.getenv("RETRIEVAL_MODE", "all"),
        help="Retrieval mode to test. Use 'all' to run all modes.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=int(os.getenv("TOP_K", "3")),
        help="Top-k documents to retrieve per query.",
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=float(os.getenv("DENSE_WEIGHT", "0.5")),
        help="Hybrid: weight for dense rank contribution.",
    )
    parser.add_argument(
        "--sparse-weight",
        type=float,
        default=float(os.getenv("SPARSE_WEIGHT", "0.5")),
        help="Hybrid: weight for sparse rank contribution.",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=int(os.getenv("RRF_K", "60")),
        help="Hybrid: RRF constant k.",
    )
    parser.add_argument(
        "--queries",
        type=str,
        default=None,
        help="Comma-separated list of test queries. If omitted, defaults to 'tire pressure,oil change interval'.",
    )

    # Multi-Query Fusion flags (defaults from env to align with main)
    mq_default = (os.getenv("MULTI_QUERY", "").strip().lower() in ("1", "true", "yes"))
    mq_rrf_env = os.getenv("MULTI_QUERY_RRF_K", None)
    mq_rrf_default = int(mq_rrf_env) if mq_rrf_env not in (None, "") else None

    parser.add_argument(
        "--multi-query",
        action="store_true",
        default=mq_default,
        help="Enable multi-query expansion/fusion. Default from env MULTI_QUERY in ('1','true','yes').",
    )
    parser.add_argument(
        "--num-query-variants",
        type=int,
        default=int(os.getenv("NUM_QUERY_VARIANTS", "4")),
        help="Number of expansion variants when multi-query is enabled.",
    )
    parser.add_argument(
        "--expansion-method",
        choices=["heuristic", "llm"],
        default=os.getenv("EXPANSION_METHOD", "heuristic"),
        help="Expansion method for multi-query; this harness does not instantiate LLMs.",
    )
    parser.add_argument(
        "--multi-query-rrf-k",
        type=int,
        default=mq_rrf_default,
        help="Optional override for RRF K used in multi-query fusion (default falls back to --rrf-k).",
    )

    return parser.parse_args()


def _format_row(rank: int, doc: Document) -> str:
    meta = getattr(doc, "metadata", {}) or {}
    ident = derive_identity(doc)
    page = meta.get("page", "")
    source = meta.get("source", "")
    s_val = meta.get("sparse_score")
    rrf_val = meta.get("hybrid_rrf_score")
    mq_rrf_val = meta.get("multi_query_rrf_score")
    try:
        s_score_str = f"{float(s_val):.4f}"
    except Exception:
        s_score_str = "-" if s_val is None else str(s_val)
    try:
        rrf_str = f"{float(rrf_val):.6f}"
    except Exception:
        rrf_str = "-" if rrf_val is None else str(rrf_val)
    try:
        mq_rrf_str = f"{float(mq_rrf_val):.6f}"
    except Exception:
        mq_rrf_str = "-" if mq_rrf_val is None else str(mq_rrf_val)
    return (
        f"{rank:>3} | id={ident} | page={page or '-':>3} | source={source or '-'}"
        f" | sparse_score={s_score_str} | hybrid_rrf_score={rrf_str} | multi_query_rrf_score={mq_rrf_str}"
    )


def _print_header(mode: str, query: str, k: int) -> None:
    print(f"\n[Mode={mode}] Query='{query}' (k={k})")
    print("---- results ----")


def run_for_mode(
    manager: VectorStoreManager,
    mode: str,
    k: int,
    dense_weight: float,
    sparse_weight: float,
    rrf_k: int,
    queries: List[str],
    multi_query: bool,
    num_query_variants: int,
    expansion_method: str,
    multi_query_rrf_k: Optional[int],
) -> None:
    # Log resolved Multi-Query config for this mode
    effective_mq_rrf_k = multi_query_rrf_k if multi_query_rrf_k is not None else rrf_k
    print(
        f"[Mode={mode}] MQ config: multi_query={bool(multi_query)} | "
        f"num_query_variants={int(num_query_variants)} | expansion_method={expansion_method} | "
        f"multi_query_rrf_k={multi_query_rrf_k if multi_query_rrf_k is not None else f'(fallback to rrf_k={rrf_k})'}"
    )
    if bool(multi_query) and int(num_query_variants) <= 1:
        print("Warning: multi-query enabled but num_query_variants <= 1; fusion is disabled (no-op).")

    retriever = manager.build_retriever(
        mode=mode,
        k=k,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        rrf_k=rrf_k,
        multi_query=multi_query,
        num_query_variants=num_query_variants,
        expansion_method=expansion_method,
        expansion_llm=None,  # retrieval-only harness; do not instantiate LLMs here
        multi_query_rrf_k=multi_query_rrf_k,
    )
    for q in queries:
        _print_header(mode, q, k)
        try:
            results = retriever.invoke(q)
        except Exception as e:
            print(f"Error invoking retriever for mode={mode}: {e}")
            results = []
        if not results:
            print("0 results")
        else:
            for i, doc in enumerate(results, start=1):
                print(_format_row(i, doc))
        # Assertions
        assert len(results) <= int(k), f"[Invariant violated] Expected <= {k} results, got {len(results)} for mode={mode}, query='{q}'"
        seen: set[str] = set()
        for doc in results:
            did = derive_identity(doc)
            if did in seen:
                raise AssertionError(f"[Invariant violated] Duplicate identity '{did}' for mode={mode}, query='{q}'")
            seen.add(did)


def main() -> int:
    """
    CLI-driven sanity test harness for retrieval modes.
    - Loads .env and initializes VectorStoreManager and index
    - Exercises modes: semantic | keyword | hybrid | all
    - For each query prints a compact result table and enforces invariants:
        * len(results) <= k
        * no duplicate identities (prefers metadata['chunk_uid'], else hash of (source, page, head))
    Exits with code 0 on success; non-zero on assertion failure.
    """
    args = parse_cli_args()
    # Resolve queries
    if args.queries:
        queries = [q.strip() for q in args.queries.split(",") if q.strip()]
    else:
        queries = ["tire pressure", "oil change interval"]
    # Determine modes to run
    modes = ["semantic", "keyword", "hybrid"] if args.mode == "all" else [args.mode]
    # Log resolved config
    print("=== Retrieval Sanity Check ===")
    print(f"Config: modes={modes} | k={args.k} | dense_weight={args.dense_weight} | sparse_weight={args.sparse_weight} | rrf_k={args.rrf_k}")
    print(f"Queries: {queries}")
    # Initialize embeddings and manager
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", http_client=httpx.Client(verify=False))
    manager = VectorStoreManager(embeddings)
    # Ensure vector DB exists/loaded
    manager.get_or_create()
    # Run checks
    for m in modes:
        run_for_mode(
            manager,
            m,
            args.k,
            args.dense_weight,
            args.sparse_weight,
            args.rrf_k,
            queries,
            args.multi_query,
            args.num_query_variants,
            args.expansion_method,
            args.multi_query_rrf_k,
        )
    print("\nAll sanity checks completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        code = main()
        raise SystemExit(code)
    except AssertionError as e:
        print(str(e))
        raise SystemExit(1)
    except Exception as e:
        # Unexpected errors should also surface non-zero to signal failure
        print(f"Unexpected error: {e}")
        raise SystemExit(2)