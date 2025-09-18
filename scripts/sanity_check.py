import os
import json
from typing import Any, Dict, List

from dotenv import load_dotenv
import httpx
from langchain_openai import OpenAIEmbeddings

# Ensure project root is on sys.path so 'src' package can be imported
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vectorstore import VectorStoreManager
from src.data_loader import DocumentLoader


def summarize_loaded_docs(docs: List[Any]) -> Dict[str, Any]:
    total = len(docs)
    with_source = sum(1 for d in docs if d.metadata.get("source"))
    with_heading_path = sum(1 for d in docs if d.metadata.get("heading_path"))
    with_chunk_uid = sum(1 for d in docs if d.metadata.get("chunk_uid"))
    return {
        "total_loaded_docs": total,
        "with_source": with_source,
        "with_heading_path": with_heading_path,
        "with_chunk_uid": with_chunk_uid,
    }


def preview_metadatas(metadatas: List[Dict[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
    out = []
    for m in metadatas[:limit]:
        # Keep only a few interesting fields for a readable preview
        out.append({
            "source": m.get("source"),
            "heading_path": m.get("heading_path"),
            "heading": m.get("heading"),
            "section_level": m.get("section_level"),
            "page_start": m.get("page_start"),
            "page_end": m.get("page_end"),
            "chunk_uid": m.get("chunk_uid"),
        })
    return out


def main():
    """
    Sanity check for hierarchical chunking integration.

    It reports:
      - Counts for chunks present in the vector store (ids, metadatas)
      - Coverage of hierarchical metadata (heading_path, chunk_uid)
      - Sample metadata preview
      - Quick check on freshly loaded docs from the loader
    """
    load_dotenv()

    print("=== Sanity Check: Hierarchical Chunking & Vector Store ===")
    use_hierarchical = os.getenv("HIERARCHICAL_CHUNKING", "true").lower() == "true"
    print(f"HIERARCHICAL_CHUNKING={use_hierarchical}")

    # Ensure vector store is available (does not force rebuild)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", http_client=httpx.Client(verify=False))
    manager = VectorStoreManager(embeddings)
    vectorstore = manager.get_or_create()

    # Inspect vectorstore contents
    store_info = vectorstore.get()
    ids = store_info.get("ids") or []
    metadatas = store_info.get("metadatas") or []

    total_chunks = len(ids)
    with_chunk_uid = sum(1 for m in metadatas if isinstance(m, dict) and m.get("chunk_uid"))
    with_heading_path = sum(1 for m in metadatas if isinstance(m, dict) and m.get("heading_path"))

    print("\n-- Vector Store Summary --")
    print(f"Total chunks in collection: {total_chunks}")
    print(f"Chunks with chunk_uid: {with_chunk_uid}")
    print(f"Chunks with heading_path: {with_heading_path}")

    # Show a small preview
    print("\n-- Sample Metadata Preview --")
    preview = preview_metadatas(metadatas, limit=3)
    print(json.dumps(preview, indent=2))

    # Additionally, examine what the loader currently produces (without touching the store)
    print("\n-- Loader Output Snapshot (not persisted) --")
    loader = DocumentLoader()
    loaded_docs = loader.load_documents()
    snap = summarize_loaded_docs(loaded_docs)
    print(json.dumps(snap, indent=2))

    print("\nSanity check complete.")
    print("If hierarchical fields are mostly zero, ensure HIERARCHICAL_CHUNKING=true and REBUILD_VDB=true for a one-time reindex.")


if __name__ == "__main__":
    main()