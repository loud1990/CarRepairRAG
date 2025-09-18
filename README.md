# Car Repair RAG Chatbot Setup Guide

This guide will walk you through setting up and running the Car Repair RAG Chatbot.

## Prerequisites

Before you begin, ensure you have Python installed on your system.

## Setup Instructions

Follow these steps to get the chatbot up and running:

### Step 1: Configure your OpenAI API Key

Create a file named `.env` in the root directory of your project. Add your OpenAI API key to this file in the following format:

```
OPENAI_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your actual OpenAI API key.

### Step 2: Install Dependencies

Open your terminal or command prompt, navigate to the root directory of this project, and run the following command to install the required Python packages:

```bash
pip install -r requirements.txt
```

### Step 3: Run the Chatbot

The chatbot is now modular. Use main.py as the entry point.

```bash
python main.py
```

chatbot.py is legacy; redirect to main.py.

The chatbot should now be running and ready to use.

## Project Structure

src/ contains modules (data_loader.py for PDF processing, vectorstore.py for ChromaDB management, agent.py for LangGraph RAG with memory). chroma_db/ persists the vectorstore (ignored in Git).

## Adding Documents

To add more PDFs (e.g., additional manuals), place them in the pdfs/ directory. Run `python main.py` (default: incremental add with dedup by source). For full rebuild (e.g., after changes), set env var REBUILD_VDB=true: `set REBUILD_VDB=true && python main.py` (Windows) or `export REBUILD_VDB=true && python main.py` (Unix).

## Chat Memory

The chatbot now supports conversation memory: History (last 10 messages) persists in chat_history_default.json for contextual follow-ups across runs. Type 'quit' to exit; history saves automatically.

## Usage Example

Example query: "What is the oil change interval for the Corvette?" Follow-up: "How about for tires?" (uses memory context).

## Notes

Activate your virtual env (.carrepairenv ignored in Git). First run creates vectorstore (may take time for embeddings). Subsequent runs load fast. For development, check .gitignore for generated files (chroma_db/, chat_history*.json).

At any time, you may type `quit` and hit enter to exit the chatbot.

Happy repairing!

## Hierarchical Chunking (Default: ON)

The ingestion pipeline now supports hierarchical chunking to preserve document structure (titles → sections → paragraphs). Chunks carry enriched metadata like heading_path, section_level, parent_id, and stable chunk_uid for reliable upserts.

What changes:
- Ingestion: Page-level PDFs are parsed and converted into structured section trees, then chunked by token windows.
- Metadata: Each chunk includes breadcrumb context (e.g., "Chapter 2 > Engine > Oil System") and a unique chunk_uid for deduplication/upserts.
- Retrieval formatting: Answers display heading context and page ranges when available.

### Flags and Configuration

In your `.env` set:
```
HIERARCHICAL_CHUNKING=true
REBUILD_VDB=false
```

Notes:
- HIERARCHICAL_CHUNKING controls whether hierarchical chunking is used. Default is true if not set.
- Switching this flag typically requires re-building the vector DB to avoid mixing old/new metadata schemas.

### Reindex / Rebuild the Vector Store

Run a one-time rebuild after enabling or disabling hierarchical chunking, or when adding/removing many PDFs.

Windows (cmd.exe):
```
set REBUILD_VDB=true && python main.py
```

macOS/Linux (bash/zsh):
```
export REBUILD_VDB=true && python main.py
```

This will:
- Delete and recreate the Chroma collection in ./chroma_db
- Re-extract chunks with hierarchical metadata (if enabled)
- Re-embed chunks and persist the collection

Subsequent runs can omit REBUILD_VDB to perform incremental additions.

### Expected Retrieval Output

The agent will show context above each chunk:
```
Result 1: Chapter 2 > Engine > Oil System | 2024CorvetteOwnersManual.pdf | pp. 45-46 | level 2
<chunk text...>
```

If a result spans a single page, you may see:
```
... | p. 45 | level 2
```

### Sanity Check Script

A quick sanity script is available to validate chunk counts and metadata integrity (ensures heading_path and chunk_uid are present):

Run:
```
python scripts/sanity_check.py
```

It prints:
- Total chunks
- Number with heading_path
- Number with chunk_uid
- Sample chunk metadata preview

If you disable hierarchical chunking (`HIERARCHICAL_CHUNKING=false`) and rebuild, the script will reflect legacy metadata (fewer enriched fields).

## Hybrid Search (Dense + Sparse)

### What is Hybrid Search
- Combines dense semantic retrieval with sparse keyword (BM25) search to balance intent understanding with exact term matching.
- Fusion method: Weighted Reciprocal Rank Fusion (RRF)
  - score(d) = dense_weight * 1/(rrf_k + rank_dense(d)) + sparse_weight * 1/(rrf_k + rank_sparse(d))
  - Implemented in [HybridRetriever.invoke()](src/vectorstore.py:266); constructed by [VectorStoreManager.build_retriever()](src/vectorstore.py:188).

### Dependencies and Setup
- BM25 dependency is included in [requirements.txt](requirements.txt) (rank-bm25).
- Install:
  ```
  pip install -r requirements.txt
  ```
- Create/update `.env`:
  ```
  OPENAI_API_KEY=your_api_key_here
  RETRIEVAL_MODE=hybrid
  TOP_K=3
  DENSE_WEIGHT=0.5
  SPARSE_WEIGHT=0.5
  RRF_K=60
  ```

### Runtime Toggles (CLI and Env)
- Flags parsed in [main._parse_retrieval_args()](main.py:165):
  - --retrieval-mode {semantic,keyword,hybrid}
  - --k INT
  - --dense-weight FLOAT
  - --sparse-weight FLOAT
  - --rrf-k INT
- Windows-friendly examples:
  ```
  python main.py --retrieval-mode hybrid --k 5
  python main.py --retrieval-mode keyword --k 5
  python main.py --retrieval-mode semantic --k 3
  python main.py --retrieval-mode hybrid --k 5 --dense-weight 0.7 --sparse-weight 0.3 --rrf-k 60
  ```
- Mapping:
  - keyword = strict keyword BM25 search (sparse)
  - semantic = intent-focused dense vectors
  - default = hybrid
- The retriever is constructed at [main.py](main.py:218) via [VectorStoreManager.build_retriever()](src/vectorstore.py:188) and passed unchanged into [run_agent()](src/agent.py:94).

### Architecture Pointers
- Builders and hybrid:
  - [VectorStoreManager.get_dense_retriever()](src/vectorstore.py:169)
  - [VectorStoreManager.get_sparse_retriever()](src/vectorstore.py:148)
  - [VectorStoreManager.build_retriever()](src/vectorstore.py:188)
  - [HybridRetriever.invoke()](src/vectorstore.py:266)
- Identity helper (stable IDs for dedupe/tests):
  - [VectorStoreManager._doc_identity()](src/vectorstore.py:31)
- Agent integration (retriever is injected and used):
  - [retriever_tool()](src/agent.py:17)
  - [run_agent()](src/agent.py:94)
- CLI parse and call site:
  - [main._parse_retrieval_args()](main.py:165)
  - [main.py](main.py:218)

### Retrieval Sanity Checks
- Harness: [scripts/sanity_check.py](scripts/sanity_check.py)
- Run:
  ```
  python scripts/sanity_check.py
  python scripts/sanity_check.py --mode hybrid --k 5
  python scripts/sanity_check.py --mode keyword --k 5
  python scripts/sanity_check.py --mode semantic --k 3
  python scripts/sanity_check.py --mode all --queries "tire pressure,oil change interval"
  ```
- Validates:
  - Results per mode, up to k.
  - No duplicate chunk identities.
  - Displays sparse_score and hybrid_rrf_score when present.

### Troubleshooting and Notes
- Ensure OPENAI_API_KEY is set in `.env`.
- Rebuild vector DB if needed (see Reindex/Rebuild section above). Dense index is loaded/built at runtime by [VectorStoreManager.get_or_create()](src/vectorstore.py:46); sparse retriever is built on demand by [VectorStoreManager.get_sparse_retriever()](src/vectorstore.py:148).
- Persistence: Chroma collection persisted under `./chroma_db` (default in [VectorStoreManager.__init__()](src/vectorstore.py:13)).
- Hybrid Search is drop-in; agent logic is unchanged and continues to use the injected retriever via [retriever_tool()](src/agent.py:17) and [run_agent()](src/agent.py:94).
