"""
Sparse BM25 index and retriever wrapper.

This module implements a lightweight sparse keyword search using BM25 over a set
of LangChain Documents. It provides a "retriever-like" wrapper exposing
invoke(query: str) -> List[Document].
"""

import re
from typing import Callable, List, Optional, Sequence

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

TOKEN_REGEX = re.compile(r"\w+", flags=re.UNICODE)


def default_tokenize(text: str) -> List[str]:
    """
    Lowercase + regex word-splitting tokenizer.
    Uses r"\w+" on lowercased text to produce tokens.
    """
    if not text:
        return []
    return TOKEN_REGEX.findall(text.lower())


class BM25SparseIndex:
    """
    BM25 index over a collection of LangChain Documents.
    - Stores references to the original Document objects (no cloning).
    - Does not mutate metadata keys used for identity. Only writes 'sparse_score'.
    """

    def __init__(self, documents: Sequence[Document], tokenizer: Optional[Callable[[str], List[str]]] = None) -> None:
        self._docs: List[Document] = list(documents or [])
        self._tokenize = tokenizer or default_tokenize

        # Pre-tokenize corpus
        self._corpus_tokens: List[List[str]] = [
            self._tokenize(getattr(d, "page_content", "") or "") for d in self._docs
        ]

        # If there are no documents or the entire corpus is effectively empty, avoid constructing BM25
        if not self._docs or sum(len(t) for t in self._corpus_tokens) == 0:
            self._bm25: Optional[BM25Okapi] = None
        else:
            self._bm25 = BM25Okapi(self._corpus_tokens)

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Execute a sparse BM25 search.

        Returns the original Document objects. Attaches a float score in
        doc.metadata['sparse_score'] for the current query.
        """
        if not self._docs or not self._bm25:
            return []

        q_tokens = self._tokenize(query or "")
        if not q_tokens:
            return []

        scores = self._bm25.get_scores(q_tokens)
        # Rank by score descending and select top_k
        k = max(0, min(int(top_k or 0), len(self._docs)))
        if k == 0:
            return []

        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results: List[Document] = []
        for idx in ranked_indices:
            doc = self._docs[idx]  # reference to original object
            # Only attach score; do not mutate any existing identity keys
            try:
                # Ensure metadata exists
                if not hasattr(doc, "metadata") or doc.metadata is None:
                    # LangChain Document always has metadata dict, but guard anyway
                    doc.metadata = {}
                doc.metadata["sparse_score"] = float(scores[idx])
            except Exception:
                # Be resilient: if metadata is not dict-like, skip attaching
                pass
            results.append(doc)

        return results


class BM25SparseRetriever:
    """
    Thin 'retriever-like' wrapper exposing invoke(query: str) -> List[Document].
    Top-k is fixed at construction time.
    """

    def __init__(self, index: BM25SparseIndex, k: int = 5) -> None:
        self.index = index
        self.k = int(k) if k is not None else 5

    def invoke(self, query: str) -> List[Document]:
        return self.index.retrieve(query, top_k=self.k)

    # Allow call shorthand
    __call__ = invoke


def build_sparse_retriever(
    documents: Sequence[Document],
    top_k: int = 5,
    tokenizer: Optional[Callable[[str], List[str]]] = None,
) -> BM25SparseRetriever:
    """
    Convenience builder that constructs a BM25 index and returns a retriever wrapper.
    """
    index = BM25SparseIndex(documents, tokenizer=tokenizer)
    return BM25SparseRetriever(index, k=top_k)


__all__ = ["BM25SparseIndex", "BM25SparseRetriever", "build_sparse_retriever"]