import os
import hashlib
from typing import Optional, List, Literal, Any
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.data_loader import DocumentLoader
from src.sparse_index import build_sparse_retriever
from src.query_expander import QueryExpander
from src.fusion import MultiQueryFusionRetriever
from langsmith import traceable
import chromadb


class VectorStoreManager:
    def __init__(self, embeddings=None, persist_directory="./chroma_db", collection_name="car_repair_docs"):
        self.embeddings = embeddings or OpenAIEmbeddings(model="text-embedding-3-small")
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.vectorstore = None
        # Cache for a process-local sparse (BM25) retriever instance
        self._sparse_retriever = None

    def _chunk_uid(self, doc) -> Optional[str]:
        return doc.metadata.get("chunk_uid")
    
    def _fallback_id(self, doc) -> str:
        src = doc.metadata.get("source", "")
        pg = str(doc.metadata.get("page", ""))
        sample = (doc.page_content or "")[:200]
        raw = f"{src}|{pg}|{sample}"
        return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()
    
    def _doc_identity(self, doc) -> str:
        """
        Private helper to derive a stable document identity.
        Prefer metadata['chunk_uid'] if present; otherwise fall back to a
        deterministic hash of (source, page, content sample).
        """
        cid = None
        try:
            cid = self._chunk_uid(doc)
        except Exception:
            cid = None
        if cid:
            return str(cid)
        return self._fallback_id(doc)

    def get_or_create(self):
        rebuild = os.getenv("REBUILD_VDB", "false").lower() == "true"
        if rebuild:
            self.rebuild()
            return self.vectorstore

        client = chromadb.PersistentClient(path=self.persist_directory)
        try:
            collection = client.get_collection(self.collection_name)
            self.vectorstore = Chroma(
                client=client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            )
            print("Loaded existing ChromaDB vector store!")
            # Incremental add after successful load
            loader = DocumentLoader()
            all_docs = loader.load_documents()

            store_info = self.vectorstore.get()
            existing_ids = set(store_info.get("ids") or [])
            existing_metas = store_info.get("metadatas") or []
            existing_chunk_uids = {m.get("chunk_uid") for m in existing_metas if isinstance(m, dict) and m.get("chunk_uid")}
            existing_identifiers = existing_ids | existing_chunk_uids

            has_chunk_uid = any("chunk_uid" in d.metadata for d in all_docs)

            if has_chunk_uid:
                docs_and_ids = [(d, self._chunk_uid(d)) for d in all_docs if self._chunk_uid(d)]
                new = [(d, cid) for (d, cid) in docs_and_ids if cid not in existing_identifiers]
                if new:
                    new_docs = [d for d, _ in new]
                    new_ids = [cid for _, cid in new]
                    self.vectorstore.add_documents(new_docs, ids=new_ids)
                    print(f"Added {len(new_docs)} new hierarchical chunks from updated PDFs.")
            else:
                existing_sources = set(meta.get('source', '') for meta in existing_metas)
                new_docs = [d for d in all_docs if d.metadata.get('source', '') not in existing_sources]
                if new_docs:
                    self.vectorstore.add_documents(new_docs)
                    print(f"Added {len(new_docs)} new chunks from updated PDFs.")
            return self.vectorstore
        except Exception as e:
            print(f"Error loading collection: {e}. Creating new one...")
            loader = DocumentLoader()
            docs = loader.load_documents()
            ids = [d.metadata.get("chunk_uid") for d in docs]
            if ids and all(ids):
                self.vectorstore = Chroma.from_documents(
                    documents=docs,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory,
                    collection_name=self.collection_name,
                    ids=ids
                )
            else:
                self.vectorstore = Chroma.from_documents(
                    documents=docs,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory,
                    collection_name=self.collection_name
                )
            print("Created ChromaDB vector store!")
            return self.vectorstore

    def add_documents(self, docs):
        if self.vectorstore:
            ids = [d.metadata.get("chunk_uid") for d in docs]
            if ids and all(ids):
                self.vectorstore.add_documents(docs, ids=ids)
            else:
                self.vectorstore.add_documents(docs)
        else:
            raise ValueError("Vectorstore not initialized")

    def rebuild(self):
        client = chromadb.PersistentClient(path=self.persist_directory)
        try:
            client.delete_collection(self.collection_name)
            print("Deleted existing collection.")
        except Exception:
            print("No existing collection to delete.")
        loader = DocumentLoader()
        docs = loader.load_documents()
        ids = [d.metadata.get("chunk_uid") for d in docs]
        if ids and all(ids):
            self.vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
                ids=ids
            )
        else:
            self.vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )
        print("Rebuilt ChromaDB vector store!")

    @traceable(name="sparse_retrieval")
    def get_sparse_retriever(self, k: int = 5):
        """
        Build (once) and return a sparse-only BM25 retriever over the same document chunks
        used by the vector index. The retriever exposes invoke(query: str) -> List[Document].
        Cached on the manager instance to avoid reconstruction.
        """
        if getattr(self, "_sparse_retriever", None) is not None:
            return self._sparse_retriever
    
        # Load documents via the existing loader path
        try:
            loader = DocumentLoader()
            docs = loader.load_documents()
        except Exception as e:
            # Fall back to an empty retriever if documents cannot be loaded
            print(f"Warning: could not load documents for sparse retriever: {e}")
            docs = []
    
        self._sparse_retriever = build_sparse_retriever(docs, top_k=int(k) if k is not None else 5)
        return self._sparse_retriever

    @traceable(name="dense_retrieval")
    def get_dense_retriever(self, k: int = 3, search_type: str = "similarity"):
        """
        Build and return a dense (semantic) retriever backed by the Chroma vector store.
        
        Parameters:
            k: number of documents to return
            search_type: retrieval strategy, e.g., 'similarity', 'mmr'
        
        Returns:
            A retriever-like object supporting invoke(query: str) -> List[Document].
        """
        if not self.vectorstore:
            self.get_or_create()
        # LangChain retrievers are Runnables and support .invoke
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": int(k) if k is not None else 3},
        )

    def build_retriever(
        self,
        mode: Literal["semantic", "keyword", "hybrid"] = "hybrid",
        k: int = 3,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        rrf_k: int = 60,
        multi_query: bool = False,
        num_query_variants: int = 1,
        expansion_method: Literal["llm", "heuristic"] = "heuristic",
        expansion_llm: Optional[Any] = None,
        multi_query_rrf_k: Optional[int] = None,
    ):
        """
        Create a unified retriever according to the requested mode.
        
        - semantic: dense retriever via Chroma
        - keyword: sparse BM25 retriever
        - hybrid: fuse dense + sparse via Weighted Reciprocal Rank Fusion (RRF)
        
        When multi_query is enabled and num_query_variants > 1, wraps the base retriever
        with a MultiQueryFusionRetriever that expands queries and fuses results via RRF.
        
        Returns:
            A retriever-like object exposing invoke(query: str) -> List[Document].
        """
        # Build base retriever according to mode (unchanged defaults)
        if mode == "semantic":
            base = self.get_dense_retriever(k=k)
        elif mode == "keyword":
            base = self.get_sparse_retriever(k=k)
        else:
            # Default to hybrid
            dense = self.get_dense_retriever(k=k)
            sparse = self.get_sparse_retriever(k=k)
            base = HybridRetriever(
                dense_retriever=dense,
                sparse_retriever=sparse,
                top_k=int(k) if k is not None else 3,
                dense_weight=float(dense_weight),
                sparse_weight=float(sparse_weight),
                rrf_k=int(rrf_k) if rrf_k is not None else 60,
            )

        # Optionally wrap with multi-query fusion
        if bool(multi_query) and int(num_query_variants) > 1:
            expander = QueryExpander(llm=expansion_llm, default_method=expansion_method)
            return MultiQueryFusionRetriever(
                base_retriever=base,
                expander=expander,
                top_k=int(k) if k is not None else 3,
                n_variants=int(num_query_variants),
                rrf_k=int(multi_query_rrf_k) if multi_query_rrf_k is not None else (int(rrf_k) if rrf_k is not None else 60),
                expansion_method=expansion_method,
            )

        return base


class HybridRetriever:
    """
    Hybrid retriever that fuses dense and sparse candidates with Weighted Reciprocal Rank Fusion (RRF).

    score(d) = dense_weight * 1/(rrf_k + rank_dense(d)) + sparse_weight * 1/(rrf_k + rank_sparse(d))
    Ranks are 1-based. Missing ranks are skipped.
    """

    def __init__(
        self,
        dense_retriever,
        sparse_retriever,
        top_k: int = 3,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        rrf_k: int = 60,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.top_k = int(top_k) if top_k is not None else 3
        self.dense_weight = float(dense_weight)
        self.sparse_weight = float(sparse_weight)
        self.rrf_k = int(rrf_k) if rrf_k is not None else 60

    def _identity(self, doc: Document) -> str:
        """
        Derive stable identity, preferring metadata['chunk_uid'].
        Falls back to stable hash over (source, page, head content).
        """
        try:
            meta = getattr(doc, "metadata", {}) or {}
        except Exception:
            meta = {}
        cid = meta.get("chunk_uid")
        if cid:
            return str(cid)
        src = meta.get("source", "")
        pg = str(meta.get("page", ""))
        sample = (getattr(doc, "page_content", "") or "")[:200]
        raw = f"{src}|{pg}|{sample}"
        return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()

    @traceable(name="hybrid_retrieval")
    def invoke(self, query: str) -> List[Document]:
        """Hybrid retrieval with RRF fusion."""
        # Retrieve candidates
        dense_list: List[Document] = []
        sparse_list: List[Document] = []

        try:
            res = self.dense_retriever.invoke(query)
            if isinstance(res, list):
                dense_list = res
        except Exception as e:
            print(f"HybridRetriever: dense retriever failed: {e}")

        try:
            res = self.sparse_retriever.invoke(query)
            if isinstance(res, list):
                sparse_list = res
        except Exception as e:
            print(f"HybridRetriever: sparse retriever failed: {e}")

        # Index by identity and store ranks
        dense_by_id = {}
        sparse_by_id = {}
        dense_ranks = {}
        sparse_ranks = {}

        for rank, d in enumerate(dense_list, start=1):
            did = self._identity(d)
            dense_ranks[did] = rank
            if did not in dense_by_id:
                dense_by_id[did] = d

        for rank, d in enumerate(sparse_list, start=1):
            sid = self._identity(d)
            sparse_ranks[sid] = rank
            if sid not in sparse_by_id:
                sparse_by_id[sid] = d

        all_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())

        # Weighted RRF scoring
        scores = {}
        for did in all_ids:
            score = 0.0
            if did in dense_ranks:
                score += self.dense_weight * (1.0 / (self.rrf_k + dense_ranks[did]))
            if did in sparse_ranks:
                score += self.sparse_weight * (1.0 / (self.rrf_k + sparse_ranks[did]))
            scores[did] = score

        # Order by fused score descending
        ordered_ids = sorted(all_ids, key=lambda x: scores.get(x, 0.0), reverse=True)

        results: List[Document] = []
        for did in ordered_ids[: self.top_k]:
            # Prefer dense doc if present, else sparse
            doc = dense_by_id.get(did) or sparse_by_id.get(did)
            if not hasattr(doc, "metadata") or doc.metadata is None:
                try:
                    doc.metadata = {}
                except Exception:
                    pass

            # Preserve any existing sparse_score if available from the sparse hit
            sdoc = sparse_by_id.get(did)
            try:
                if sdoc is not None:
                    s_meta = getattr(sdoc, "metadata", {}) or {}
                    if "sparse_score" in s_meta and (not hasattr(doc, "metadata") or "sparse_score" not in doc.metadata):
                        doc.metadata["sparse_score"] = float(s_meta["sparse_score"])
            except Exception:
                pass

            # Attach hybrid fused score
            try:
                doc.metadata["hybrid_rrf_score"] = float(scores[did])
            except Exception:
                pass

            results.append(doc)

        return results

    # Allow call shorthand
    __call__ = invoke