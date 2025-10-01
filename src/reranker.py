"""
Cross-encoder re-ranker for improving retrieval quality while reducing token count.

This module provides a re-ranking layer that:
1. Retrieves a larger set of candidate documents (initial_k)
2. Re-ranks them using a cross-encoder model for semantic relevance
3. Returns only the top_n most relevant documents (top_n < initial_k)

This allows for better quality results with fewer tokens by:
- Casting a wider net initially (high recall)
- Then selecting the most relevant subset (high precision)
"""

from typing import List, Optional
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun


class CrossEncoderReranker(BaseRetriever):
    """
    A retriever wrapper that re-ranks results using a cross-encoder model.
    
    This implementation wraps any base retriever and adds semantic re-ranking
    to improve result quality while potentially reducing the number of returned documents.
    
    Attributes:
        base_retriever: The underlying retriever to fetch initial candidates
        model_name: Name of the cross-encoder model to use for re-ranking
        initial_k: Number of initial documents to retrieve (default: 10)
        top_n: Number of top documents to return after re-ranking (default: 3)
    """
    
    base_retriever: BaseRetriever
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    initial_k: int = 10
    top_n: int = 3
    _model: Optional[object] = None
    
    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"
    
    def __init__(self, **data):
        super().__init__(**data)
        # Lazy load the cross-encoder model
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for CrossEncoderReranker. "
                    "Install it with: pip install sentence-transformers"
                )
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Retrieve and re-rank documents.
        
        Process:
        1. Update base retriever to fetch initial_k documents
        2. Get candidates from base retriever
        3. Score each candidate using cross-encoder
        4. Sort by score and return top_n
        
        Args:
            query: The search query
            run_manager: Optional callback manager
            
        Returns:
            List of top_n re-ranked documents
        """
        # Temporarily update base retriever's k parameter if it has one
        original_k = None
        if hasattr(self.base_retriever, 'search_kwargs'):
            original_k = self.base_retriever.search_kwargs.get('k')
            self.base_retriever.search_kwargs['k'] = self.initial_k
        
        # Get initial candidates
        candidates = self.base_retriever.get_relevant_documents(query)
        
        # Restore original k
        if original_k is not None and hasattr(self.base_retriever, 'search_kwargs'):
            self.base_retriever.search_kwargs['k'] = original_k
        
        # If we got fewer candidates than top_n, just return them
        if len(candidates) <= self.top_n:
            return candidates
        
        # Prepare query-document pairs for cross-encoder
        pairs = [[query, doc.page_content] for doc in candidates]
        
        # Score all pairs
        scores = self._model.predict(pairs)
        
        # Create list of (score, document) tuples
        scored_docs = list(zip(scores, candidates))
        
        # Sort by score (descending) and take top_n
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Return just the documents (without scores)
        return [doc for _, doc in scored_docs[:self.top_n]]


def create_reranker(
    base_retriever: BaseRetriever,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    initial_k: int = 10,
    top_n: int = 3
) -> CrossEncoderReranker:
    """
    Convenience function to create a CrossEncoderReranker.
    
    Args:
        base_retriever: The base retriever to wrap
        model_name: Cross-encoder model to use (default: ms-marco-MiniLM-L-6-v2)
        initial_k: Number of initial documents to retrieve (default: 10)
        top_n: Number of documents to return after re-ranking (default: 3)
        
    Returns:
        Configured CrossEncoderReranker instance
        
    Example:
        >>> from langchain.vectorstores import FAISS
        >>> vectorstore = FAISS.from_documents(docs, embeddings)
        >>> base_retriever = vectorstore.as_retriever(search_kwargs={'k': 10})
        >>> reranker = create_reranker(base_retriever, initial_k=10, top_n=3)
        >>> results = reranker.get_relevant_documents("What is the oil capacity?")
    """
    return CrossEncoderReranker(
        base_retriever=base_retriever,
        model_name=model_name,
        initial_k=initial_k,
        top_n=top_n
    )