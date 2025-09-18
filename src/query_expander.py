"""
Query expansion utility for RAG-Fusion / Self-Consistency.

This module provides QueryExpander, a small dependency-light component that
creates up to N query paraphrases for retrieval. It prefers an LLM (duck-typed,
with an invoke(prompt) method) and gracefully falls back to a heuristic approach
whenever no LLM is supplied or the call fails.

No external dependencies are required; everything uses the Python standard library.
The implementation is pure and side-effect free.
"""
from typing import Any, List, Optional, Literal, Iterable, Set, Sequence
import json
import re

__all__ = ["QueryExpander"]


def _extract_json_array(text: str) -> Optional[List[str]]:
    """
    Try to extract a JSON array of strings from arbitrary text.
    Returns the list if successful, else None.
    """
    if not text:
        return None
    # Direct attempt
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if isinstance(x, (str, int, float))]
        if isinstance(obj, dict):
            # Common keys sometimes used by models
            for key in ("queries", "variations", "results", "items"):
                if key in obj and isinstance(obj[key], list):
                    return [str(x).strip() for x in obj[key] if isinstance(x, (str, int, float))]
    except Exception:
        pass
    # Look for the first [...] block
    try:
        m = re.search(r"\[.*?\]", text, flags=re.DOTALL)
        if m:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if isinstance(x, (str, int, float))]
    except Exception:
        pass
    # Strip code fences and try again
    try:
        cleaned = "\n".join(
            ln for ln in text.splitlines() if not ln.strip().startswith("```")
        )
        m = re.search(r"\[.*?\]", cleaned, flags=re.DOTALL)
        if m:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if isinstance(x, (str, int, float))]
    except Exception:
        pass
    return None


def _split_candidates(text: str) -> List[str]:
    """
    Fallback parsing: split text into candidate lines, stripping bullets and numbers.
    """
    if not text:
        return []
    lines: List[str] = []
    # Remove code fences
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith("```"):
            continue
        # Remove leading bullets/numbering like '- ', '* ', '1. ', '• '
        s = re.sub(r"^\s*(?:[-*•]+|\d+[\.)])\s*", "", s).strip()
        # Remove surrounding quotes
        s = s.strip(" '\"")
        if s:
            lines.append(s)
    # Also split any semicolon-separated items
    out: List[str] = []
    for ln in lines:
        parts = [p.strip().strip(" '\"") for p in re.split(r"[;|]", ln) if p.strip()]
        out.extend(parts if parts else [ln])
    return out


def _replace_word(text: str, old: str, new: str) -> Optional[str]:
    """
    Replace a whole-word occurrence of 'old' with 'new' (case-insensitive).
    Returns the replaced string if a change occurred, else None.
    """
    if not text or not old or not new:
        return None
    pattern = re.compile(rf"\b{re.escape(old)}\b", flags=re.IGNORECASE)
    replaced = pattern.sub(new, text)
    if replaced != text:
        return replaced
    return None


_SYNONYM_PAIRS: Sequence[tuple[str, str]] = (
    ("tire", "tyre"),
    ("manual", "guide"),
    ("oil", "lubricant"),
    ("engine", "motor"),
    ("hood", "bonnet"),
    ("trunk", "boot"),
    ("gas", "fuel"),
    ("gasoline", "petrol"),
    ("windshield", "windscreen"),
    ("wrench", "spanner"),
)

_TEMPLATES: Sequence[str] = ("how to", "what is", "guide to", "steps to")


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    """
    Deduplicate strings case-insensitively while preserving order.
    """
    seen: Set[str] = set()
    out: List[str] = []
    for it in items:
        s = (it or "").strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


class QueryExpander:
    """
    Generate query variations for retrieval using an LLM (preferred) with a heuristic fallback.

    The LLM, if provided, is expected to expose an `invoke(prompt: str) -> Any` method.
    """

    def __init__(self, llm: Optional[Any] = None, default_method: Literal["llm", "heuristic"] = "llm") -> None:
        """
        Initialize the expander.

        Args:
            llm: Optional duck-typed LLM client with an invoke(prompt) method, e.g., a LangChain Chat model.
            default_method: Which method to use if none is specified in expand(). "llm" or "heuristic".
        """
        if default_method not in ("llm", "heuristic"):
            raise ValueError("default_method must be 'llm' or 'heuristic'")
        self._llm = llm
        self._default_method: Literal["llm", "heuristic"] = default_method

    def _call_llm(self, prompt: str) -> Optional[str]:
        """
        Attempt to call the provided LLM. Returns the text content, or None on failure.
        """
        llm = self._llm
        if llm is None:
            return None
        try:
            result = llm.invoke(prompt)  # type: ignore[attr-defined]
        except Exception:
            return None
        # Extract text
        try:
            if isinstance(result, str):
                return result
            if hasattr(result, "content"):
                content = getattr(result, "content", None)
                if isinstance(content, str):
                    return content
            if isinstance(result, dict):
                for key in ("content", "text", "output_text"):
                    v = result.get(key)  # type: ignore[call-arg]
                    if isinstance(v, str):
                        return v
            # Fallback to string representation
            return str(result)
        except Exception:
            return None

    def _llm_expand(self, query: str, n: int) -> List[str]:
        """
        LLM-powered expansion. Returns zero or more candidates (not guaranteed to include the original).
        """
        n = max(0, int(n))
        if n == 0:
            return []
        prompt = (
            "You are a query expansion assistant for document retrieval.\n"
            "Given a user's query, produce diverse paraphrases/synonymized queries that preserve intent.\n"
            f"Return a JSON array with up to {n} short queries. No commentary or extra text.\n"
            "Aim for lexical diversity (synonyms, phrasing changes), avoid changing meaning.\n"
            f"Query: {query}"
        )
        text = self._call_llm(prompt)
        if not text:
            return []
        # Try to parse as JSON array
        arr = _extract_json_array(text)
        if arr is None:
            # Fallback: split into lines
            arr = _split_candidates(text)
        # Normalize, dedupe, cap
        arr = _dedupe_preserve_order([a.strip() for a in arr if isinstance(a, str) and a.strip()])
        if len(arr) > n:
            arr = arr[:n]
        return arr

    def _heuristic_expand(self, query: str, n: int) -> List[str]:
        """
        Heuristic expansion using simple templates and light synonym substitutions.
        Returns zero or more candidates (not guaranteed to include the original).
        """
        n = max(0, int(n))
        if n == 0:
            return []
        base = re.sub(r"[?.!\s]+$", "", (query or "").strip())
        q_lower = base.lower()
        candidates: List[str] = []
        # Case-normalized variant
        if q_lower and q_lower != query:
            candidates.append(q_lower)
        # Template prepends
        for t in _TEMPLATES:
            candidates.append(f"{t} {q_lower}")
        # Synonym swaps on original and lower-case forms
        for a, b in _SYNONYM_PAIRS:
            rep1 = _replace_word(base, a, b)
            if rep1:
                candidates.append(rep1)
            rep2 = _replace_word(base, b, a)
            if rep2:
                candidates.append(rep2)
            rep3 = _replace_word(q_lower, a, b)
            if rep3:
                candidates.append(rep3)
            rep4 = _replace_word(q_lower, b, a)
            if rep4:
                candidates.append(rep4)
        # Combine templates with synonym variants (lower-cased forms)
        lower_syns = [c for c in candidates if c == c.lower()]
        for t in _TEMPLATES:
            for v in lower_syns:
                candidates.append(f"{t} {v}")
        # Deduplicate and cap
        candidates = _dedupe_preserve_order([c for c in candidates if c and c.strip()])
        if len(candidates) > n:
            candidates = candidates[:n]
        return candidates

    def expand(self, query: str, n: int = 4, method: Optional[Literal["llm", "heuristic"]] = None) -> List[str]:
        """
        Generate up to n unique query variants, including the original as the first element.

        Args:
            query: The input query to expand.
            n: Maximum number of variants to return (including the original).
            method: "llm", "heuristic", or None to use the default configured method.

        Returns:
            A list of up to n unique strings, with the original query first.
        """
        n = max(0, int(n))
        if n == 0:
            return []
        chosen: Literal["llm", "heuristic"] = method or self._default_method
        # Collect expansions
        expansions: List[str]
        if chosen == "llm":
            expansions = self._llm_expand(query, max(0, n - 1))
            # Fallback if LLM unavailable or produced nothing
            if not expansions:
                expansions = self._heuristic_expand(query, max(0, n - 1))
        else:
            expansions = self._heuristic_expand(query, max(0, n - 1))
        # Build final list with original first, deduped and capped
        final: List[str] = []
        seen: Set[str] = set()

        def add(item: str) -> None:
            s = (item or "").strip()
            if not s:
                return
            key = s.lower()
            if key in seen:
                return
            seen.add(key)
            final.append(s)

        add(query)  # original first
        for cand in expansions:
            if len(final) >= n:
                break
            add(cand)
        return final