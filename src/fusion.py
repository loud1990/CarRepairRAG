"""
Multi-query fusion retriever module.

Provides MultiQueryFusionRetriever wrapper that expands a user query into N variants
using a duck-typed QueryExpander and fuses results from a base retriever using
Reciprocal Rank Fusion (RRF).

- Dependencies: standard library only (typing, hashlib)
- Types: duck-typed; no direct LangChain imports at runtime
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal, TYPE_CHECKING
import hashlib

if TYPE_CHECKING:
    # For type hints only; not imported at runtime
    from langchain_core.documents import Document  # pragma: no cover


class MultiQueryFusionRetriever:
    """
    Expand the input query into multiple variants and fuse results via RRF.

    The base_retriever must expose invoke(query: str) -> List[Document].
    The expander must expose expand(query: str, n: int, method: Literal['llm','heuristic']) -> List[str].

    Returned documents include an added metadata field: 'multi_query_rrf_score' (float).
    Identity fields (e.g., 'chunk_uid') are preserved and never mutated.
    """

    def __init__(
        self,
        base_retriever: Any,
        expander: Any,
        top_k: int = 3,
        n_variants: int = 4,
        rrf_k: int = 60,
        expansion_method: Literal["llm", "heuristic"] = "heuristic",
    ) -> None:
        self.base_retriever = base_retriever
        self.expander = expander
        self.top_k = int(top_k) if top_k is not None else 3
        self.n_variants = int(n_variants) if n_variants is not None else 1
        self.rrf_k = int(rrf_k) if rrf_k is not None else 60
        if expansion_method not in ("llm", "heuristic"):
            raise ValueError("expansion_method must be 'llm' or 'heuristic'")
        self.expansion_method: Literal["llm", "heuristic"] = expansion_method

    def _identity(self, doc: Any) -> str:
        """
        Derive a stable identity for a document.
        Prefer metadata['chunk_uid']; otherwise hash of (source, page, head content).
        """
        try:
            meta = getattr(doc, "metadata", {}) or {}
        except Exception:
            meta = {}
        cid = meta.get("chunk_uid")
        if cid:
            return str(cid)
        src = str(meta.get("source", ""))
        pg = str(meta.get("page", ""))
        sample = (getattr(doc, "page_content", "") or "")[:200]
        raw = f"{src}|{pg}|{sample}"
        return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()

    def _ensure_metadata_dict(self, doc: Any) -> None:
        try:
            if not hasattr(doc, "metadata") or getattr(doc, "metadata") is None:
                setattr(doc, "metadata", {})
            elif not isinstance(getattr(doc, "metadata"), dict):
                # Do not replace non-dict metadata; avoid mutation
                pass
        except Exception:
            pass

    def _expand_queries(self, query: str) -> List[str]:
        n = max(1, self.n_variants)
        variants: List[str] = []
        try:
            variants = self.expander.expand(query, n=n, method=self.expansion_method)  # type: ignore[attr-defined]
        except Exception:
            variants = []
        # Ensure original first and dedupe (case-insensitive), then cap to n
        final: List[str] = []
        seen = set()

        def add(s: str) -> None:
            text = (s or "").strip()
            if not text:
                return
            key = text.lower()
            if key in seen:
                return
            seen.add(key)
            final.append(text)

        add(query)
        for v in variants:
            if len(final) >= n:
                break
            add(v)
        return final

    def invoke(self, query: str) -> List["Document"]:
        """
        Execute multi-query retrieval and fuse with RRF.
        """
        # Expand queries
        queries = self._expand_queries(query)

        # Collect ranked lists from the base retriever
        ranked_lists: List[Dict[str, int]] = []
        chosen_doc_by_id: Dict[str, Any] = {}
        best_rank_by_id: Dict[str, int] = {}

        for q in queries:
            try:
                results = self.base_retriever.invoke(q)  # type: ignore[attr-defined]
            except Exception:
                results = []
            if not isinstance(results, list):
                results = []

            rank_map: Dict[str, int] = {}
            for rank, d in enumerate(results, start=1):
                did = self._identity(d)
                # record first rank if duplicate within the same list
                if did not in rank_map:
                    rank_map[did] = rank
                # choose representative doc with best (lowest) rank across lists
                prev = best_rank_by_id.get(did)
                if prev is None or rank < prev:
                    best_rank_by_id[did] = rank
                    chosen_doc_by_id[did] = d
            ranked_lists.append(rank_map)

        # Union of all ids
        all_ids = set()
        for rm in ranked_lists:
            all_ids.update(rm.keys())

        # Compute RRF scores
        scores: Dict[str, float] = {}
        k = float(self.rrf_k)
        for did in all_ids:
            s = 0.0
            for rm in ranked_lists:
                r = rm.get(did)
                if r is None:
                    continue
                s += 1.0 / (k + float(r))
            scores[did] = s

        # Order by fused score
        ordered_ids = sorted(all_ids, key=lambda x: scores.get(x, 0.0), reverse=True)

        # Build result documents
        out: List["Document"] = []
        for did in ordered_ids[: self.top_k]:
            doc = chosen_doc_by_id.get(did)
            if doc is None:
                continue
            self._ensure_metadata_dict(doc)
            try:
                # Attach fused score; avoid touching identity fields
                if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
                    doc.metadata["multi_query_rrf_score"] = float(scores.get(did, 0.0))
            except Exception:
                pass
            out.append(doc)

        return out

    # Allow call shorthand
    __call__ = invoke