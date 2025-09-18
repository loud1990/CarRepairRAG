import os
import hashlib
from typing import Optional
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from src.data_loader import DocumentLoader
import chromadb


class VectorStoreManager:
    def __init__(self, embeddings=None, persist_directory="./chroma_db", collection_name="car_repair_docs"):
        self.embeddings = embeddings or OpenAIEmbeddings(model="text-embedding-3-small")
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.vectorstore = None

    def _chunk_uid(self, doc) -> Optional[str]:
        return doc.metadata.get("chunk_uid")

    def _fallback_id(self, doc) -> str:
        src = doc.metadata.get("source", "")
        pg = str(doc.metadata.get("page", ""))
        sample = (doc.page_content or "")[:200]
        raw = f"{src}|{pg}|{sample}"
        return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()

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