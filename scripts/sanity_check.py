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
    return parser.parse_args()


def _format_row(rank: int, doc: Document) -> str:
    meta = getattr(doc, "metadata", {}) or {}
    ident = derive_identity(doc)
    page = meta.get("page", "")
    source = meta.get("source", "")
    s_val = meta.get("sparse_score")
    rrf_val = meta.get("hybrid_rrf_score")
    try:
        s_score_str = f"{float(s_val):.4f}"
    except Exception:
        s_score_str = "-" if s_val is None else str(s_val)
    try:
        rrf_str = f"{float(rrf_val):.6f}"
    except Exception:
        rrf_str = "-" if rrf_val is None else str(rrf_val)
    return f"{rank:>3} | id={ident} | page={page or '-':>3} | source={source or '-'} | sparse_score={s_score_str} | hybrid_rrf_score={rrf_str}"


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
) -> None:
    retriever = manager.build_retriever(
        mode=mode,
        k=k,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        rrf_k=rrf_k,
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
        run_for_mode(manager, m, args.k, args.dense_weight, args.sparse_weight, args.rrf_k, queries)
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