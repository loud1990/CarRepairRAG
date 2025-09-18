import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Optional
from langchain.schema import Document
from .chunking import hierarchical_chunker


class DocumentLoader:
    def __init__(
        self,
        pdf_directory: str = "pdfs",
        use_hierarchical: Optional[bool] = None,
        min_chunk_tokens: int = 200,
        max_chunk_tokens: int = 1000,
        overlap_tokens: int = 100,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.pdf_directory = pdf_directory
        # Default to env flag (on by default) if not explicitly provided
        self.use_hierarchical = (
            use_hierarchical
            if use_hierarchical is not None
            else os.getenv("HIERARCHICAL_CHUNKING", "true").lower() == "true"
        )
        # Token-based params for hierarchical mode
        self.min_chunk_tokens = min_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens
        # Char-based params for legacy splitter
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_documents(self) -> List[Document]:
        if not os.path.exists(self.pdf_directory):
            raise FileNotFoundError(f"PDF directory not found: {self.pdf_directory}")

        all_pages = []
        for filename in os.listdir(self.pdf_directory):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(self.pdf_directory, filename)
                try:
                    loader = PyPDFLoader(pdf_path)
                    pages = loader.load()
                    all_pages.extend(pages)
                    print(f"Loaded {len(pages)} pages from {filename}")
                except Exception as e:
                    print(f"Error loading PDF {filename}: {e}")

        if not all_pages:
            raise ValueError("No PDF documents were loaded from the specified directory.")

        # Hierarchical mode
        if self.use_hierarchical:
            split_docs = hierarchical_chunker(
                all_pages,
                min_chunk_tokens=self.min_chunk_tokens,
                max_chunk_tokens=self.max_chunk_tokens,
                overlap_tokens=self.overlap_tokens,
            )
            print(f"Total hierarchical chunks loaded: {len(split_docs)}")
            return split_docs

        # Legacy character-based splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        split_docs = text_splitter.split_documents(all_pages)

        print(f"Total chunks loaded: {len(split_docs)}")
        return split_docs