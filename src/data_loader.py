import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document


class DocumentLoader:
    def __init__(self, pdf_directory="pdfs"):
        self.pdf_directory = pdf_directory

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

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        split_docs = text_splitter.split_documents(all_pages)

        print(f"Total chunks loaded: {len(split_docs)}")
        return split_docs