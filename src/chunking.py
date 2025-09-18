import os
import re
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable, Dict, Any, Iterable

try:
    import tiktoken  # type: ignore
except Exception:
    tiktoken = None  # Fallback to whitespace tokenizer

from langchain.schema import Document


# =============== Data Model ===============

@dataclass
class SectionNode:
    node_id: str
    title: str
    level: int
    order_index: int
    spans: List[Tuple[int, int]] = field(default_factory=list)  # absolute (char_start, char_end) in doc_text
    children: List["SectionNode"] = field(default_factory=list)
    page_range: Optional[Tuple[int, int]] = None  # (page_start, page_end)
    parent_id: Optional[str] = None


# =============== Tokenization Utilities ===============

def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _get_encoding():
    if tiktoken is None:
        return None
    # cl100k_base works well for OpenAI embeddings incl. text-embedding-3-small
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    enc = _get_encoding()
    if enc is None:
        # naive fallback: approximate tokens by words
        return max(1, len(text.split()))
    return len(enc.encode(text))


def split_by_tokens(text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    """
    Split text into chunks using token windows with overlap. Uses tiktoken if available,
    otherwise falls back to word-based chunks.
    """
    assert max_tokens > 0
    overlap_tokens = max(0, min(overlap_tokens, max_tokens - 1))

    enc = _get_encoding()
    if enc is None:
        words = text.split()
        if not words:
            return []
        stride = max_tokens - overlap_tokens
        chunks = []
        for i in range(0, len(words), stride):
            chunk_words = words[i:i + max_tokens]
            chunks.append(" ".join(chunk_words))
        return chunks

    tokens = enc.encode(text)
    if not tokens:
        return []
    stride = max_tokens - overlap_tokens
    chunks = []
    for i in range(0, len(tokens), stride):
        window = tokens[i:i + max_tokens]
        chunks.append(enc.decode(window))
    return chunks


# =============== Heading Parsing & Tree Building ===============

DEFAULT_HEADING_PATTERNS: List[re.Pattern] = [
    # Markdown-style: #, ##, ###
    re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*$"),
    # Numeric outline: 1. 1.1 1.1.1 Title
    re.compile(r"^(?P<num>(?:\d+\.)+\d+|\d+)\s+(?P<title>.+?)\s*$"),
    # Chapter/Section
    re.compile(r"^(?P<kw>(?:chapter|section)\s+\d+)\.?\s+(?P<title>.+?)\s*$", re.IGNORECASE),
    # All-caps heuristic (short lines in CAPS treated as headings)
    re.compile(r"^(?P<caps>[A-Z0-9][A-Z0-9\s\-\:&/]{3,})$"),
]


def _infer_level_from_match(m: re.Match) -> int:
    if "hashes" in m.groupdict() and m.group("hashes"):
        return len(m.group("hashes"))
    if "num" in m.groupdict() and m.group("num"):
        # depth = count of dots + 1
        num = m.group("num")
        return min(6, num.count(".") + 1)
    if "kw" in m.groupdict() and m.group("kw"):
        # Chapter/Section => level 1 or 2
        kw = m.group("kw").lower()
        return 1 if "chapter" in kw else 2
    if "caps" in m.groupdict() and m.group("caps"):
        # treat all-caps as level 2 by default
        return 2
    return 3


def _extract_title_from_match(m: re.Match) -> str:
    if "title" in m.groupdict() and m.group("title"):
        return m.group("title").strip()
    if "kw" in m.groupdict() and m.group("kw"):
        return m.group("kw").strip()
    if "caps" in m.groupdict() and m.group("caps"):
        return m.group("caps").strip().title()
    return m.group(0).strip()


def parse_headings_on_text(
    text: str,
    heading_patterns: Optional[List[re.Pattern]] = None,
    max_heading_level: int = 6,
) -> List[Dict[str, Any]]:
    """
    Parse heading candidates from a document text.
    Returns a list of dicts: {start, end, content_start, title, level}
    - start: char index where heading line starts
    - end: char index where heading line ends (newline)
    - content_start: first char index of content after heading line end
    """
    patterns = heading_patterns or DEFAULT_HEADING_PATTERNS
    headings: List[Dict[str, Any]] = []

    # Iterate lines with accumulated char offsets
    offset = 0
    for line in text.splitlines(keepends=True):
        raw = line.rstrip("\r\n")
        for pat in patterns:
            m = pat.match(raw)
            if m:
                level = min(max_heading_level, max(1, _infer_level_from_match(m)))
                title = _extract_title_from_match(m)
                start = offset
                end = offset + len(line)
                content_start = end  # content starts after heading line
                headings.append(
                    {"start": start, "end": end, "content_start": content_start, "title": title, "level": level}
                )
                break
        offset += len(line)

    # Ensure at least one root heading if none found
    if not headings:
        headings = [{"start": 0, "end": 0, "content_start": 0, "title": "Document", "level": 1}]

    # Sort by start position
    headings.sort(key=lambda h: h["start"])
    return headings


def build_tree_for_doc(
    doc_text: str,
    headings: List[Dict[str, Any]],
    max_heading_level: int,
    doc_id: str,
    root_title: str,
) -> SectionNode:
    """
    Build a SectionNode tree and assign spans between headings to nodes.
    """
    # Build nodes in pre-order using a stack
    root = SectionNode(
        node_id=_sha1(f"{doc_id}|root"),
        title=root_title,
        level=0,
        order_index=0,
        spans=[],
        children=[],
        page_range=None,
        parent_id=None,
    )

    stack: List[SectionNode] = [root]
    order = 1
    # Prepare section boundaries
    for idx, h in enumerate(headings):
        level = min(max_heading_level, max(1, int(h["level"])))
        title = h["title"]
        node = SectionNode(
            node_id=_sha1(f"{doc_id}|{order}|{title}"),
            title=title,
            level=level,
            order_index=order,
            spans=[],
            children=[],
            page_range=None,
            parent_id=None,
        )
        order += 1
        # Pop to parent with lower level
        while stack and stack[-1].level >= level:
            stack.pop()
        parent = stack[-1] if stack else root
        node.parent_id = parent.node_id
        parent.children.append(node)
        stack.append(node)

    # Assign spans for each heading node = [content_start, next_heading_start)
    for i, h in enumerate(headings):
        start = h["content_start"]
        end = headings[i + 1]["start"] if i + 1 < len(headings) else len(doc_text)
        # Find the node created for this heading (order_index i+1)
        # We constructed nodes in order immediately after root; headings aligned with node order starting at 1.
        target_order = i + 1
        def _dfs_find(n: SectionNode) -> Optional[SectionNode]:
            if n.order_index == target_order:
                return n
            for c in n.children:
                r = _dfs_find(c)
                if r:
                    return r
            return None
        node = _dfs_find(root)
        if node:
            node.spans.append((start, max(start, end)))

    return root


def compute_heading_path(node: SectionNode, by_id: Dict[str, SectionNode]) -> str:
    parts = []
    cur: Optional[SectionNode] = node
    while cur and cur.level > 0:
        parts.append(cur.title.strip())
        cur = by_id.get(cur.parent_id) if cur.parent_id else None
    return " > ".join(reversed([p for p in parts if p]))


def char_span_to_page_range(
    span: Tuple[int, int],
    page_char_ranges: List[Tuple[int, int, int]],
) -> Tuple[Optional[int], Optional[int]]:
    """
    Map a char span (start, end) to inclusive page_start, page_end using page_char_ranges:
    page_char_ranges: list of (char_start, char_end, page_index)
    """
    start_char, end_char = span
    page_start = None
    page_end = None
    for (cs, ce, pidx) in page_char_ranges:
        if page_start is None and start_char <= ce and end_char >= cs:
            page_start = pidx
        if start_char <= ce and end_char >= cs:
            page_end = pidx
    return page_start, page_end


def flatten_hierarchy(
    root: SectionNode,
    *,
    doc_text: str,
    page_char_ranges: List[Tuple[int, int, int]],
    min_chunk_tokens: int,
    max_chunk_tokens: int,
    overlap_tokens: int,
    tokenizer: Optional[Callable[[str], int]] = None,
    base_metadata: Optional[Dict[str, Any]] = None,
    doc_id: str,
) -> List[Document]:
    """
    Produce chunk Documents from the section tree. Each node's text is chunked independently.
    """
    tokenizer = tokenizer or count_tokens
    chunks: List[Document] = []

    # Pre-build id -> node mapping for heading_path computation
    id_map: Dict[str, SectionNode] = {}

    def _dfs_build_id_map(n: SectionNode):
        id_map[n.node_id] = n
        for c in n.children:
            _dfs_build_id_map(c)

    _dfs_build_id_map(root)

    def _text_for_spans(spans: Iterable[Tuple[int, int]]) -> str:
        parts = []
        for (s, e) in spans:
            if s < e:
                parts.append(doc_text[s:e])
        return "\n".join(parts).strip()

    def _dfs(n: SectionNode):
        # Skip root for chunking; only chunk actual headings
        if n.level > 0:
            node_text = _text_for_spans(n.spans)
            if node_text:
                pieces = split_by_tokens(node_text, max_chunk_tokens, overlap_tokens)
                # Merge small tail with previous to honor min_chunk_tokens
                merged: List[str] = []
                for piece in pieces:
                    if not merged:
                        merged.append(piece)
                        continue
                    if tokenizer(piece) < min_chunk_tokens:
                        merged[-1] = (merged[-1] + "\n\n" + piece).strip()
                    else:
                        merged.append(piece)
                if not merged:
                    merged = pieces

                heading_path = compute_heading_path(n, id_map)
                page_s = page_e = None
                # Approximate page range by union of node spans
                if n.spans:
                    ps, pe = char_span_to_page_range(
                        (n.spans[0][0], n.spans[-1][1]),
                        page_char_ranges,
                    )
                    page_s, page_e = ps, pe

                for idx, text_piece in enumerate(merged):
                    meta = dict(base_metadata or {})
                    meta.update({
                        "doc_id": doc_id,
                        "node_id": n.node_id,
                        "parent_id": n.parent_id,
                        "section_level": n.level,
                        "heading": n.title,
                        "heading_path": heading_path,
                        "order_index": n.order_index,
                        "page_start": page_s,
                        "page_end": page_e,
                        "chunk_index": idx,
                    })
                    chunk_uid = f"{doc_id}:{n.node_id}:{idx}"
                    meta["chunk_uid"] = chunk_uid
                    chunks.append(Document(page_content=text_piece, metadata=meta))
        for c in n.children:
            _dfs(c)

    _dfs(root)
    return chunks


# =============== Public API ===============

def hierarchical_chunker(
    documents: List[Document],
    *,
    min_chunk_tokens: int = 200,
    max_chunk_tokens: int = 1000,
    overlap_tokens: int = 100,
    heading_patterns: Optional[List[re.Pattern]] = None,
    max_heading_level: int = 4,
    id_namespace: Optional[str] = None,
    tokenizer: Optional[Callable[[str], int]] = None,
) -> List[Document]:
    """
    Group input page-level documents by their source and apply hierarchical chunking, producing
    enriched chunks with hierarchical metadata.
    """
    if not documents:
        return []

    # Group pages by source (file path)
    grouped: Dict[str, List[Document]] = {}
    for d in documents:
        src = d.metadata.get("source", "unknown_source")
        grouped.setdefault(src, []).append(d)

    all_chunks: List[Document] = []

    for source, pages in grouped.items():
        # Sort by page index if present, else preserve order
        def _page_num(doc: Document) -> int:
            p = doc.metadata.get("page")
            try:
                return int(p)
            except Exception:
                return 0

        pages_sorted = sorted(pages, key=_page_num)

        # Concatenate text and track per-page char ranges
        texts: List[str] = []
        page_char_ranges: List[Tuple[int, int, int]] = []
        offset = 0
        for i, pg in enumerate(pages_sorted):
            content = pg.page_content or ""
            start = offset
            texts.append(content)
            offset += len(content)
            end = offset
            # add a newline separator between pages
            texts.append("\n")
            offset += 1
            page_index = _page_num(pg)
            page_char_ranges.append((start, end, page_index))

        doc_text = "".join(texts)
        base_name = os.path.basename(source)
        doc_id_seed = f"{id_namespace or ''}|{source}"
        doc_id = _sha1(doc_id_seed)

        # Parse headings
        headings = parse_headings_on_text(doc_text, heading_patterns, max_heading_level)

        # Build tree
        root = build_tree_for_doc(doc_text, headings, max_heading_level, doc_id=doc_id, root_title=base_name)

        # Flatten to chunks
        base_metadata = {"source": source}
        chunks = flatten_hierarchy(
            root,
            doc_text=doc_text,
            page_char_ranges=page_char_ranges,
            min_chunk_tokens=min_chunk_tokens,
            max_chunk_tokens=max_chunk_tokens,
            overlap_tokens=overlap_tokens,
            tokenizer=tokenizer,
            base_metadata=base_metadata,
            doc_id=doc_id,
        )
        all_chunks.extend(chunks)

    return all_chunks