"""
Document retrieval for the RAG system.
"""
import logging
import os
import numpy as np
from typing import List, Dict, Any, Tuple, Union

from src.rag.chunker import DocumentChunker
from src.rag.embedder import TextEmbedder
from src.rag.ranker import SnippetRanker

logger = logging.getLogger(__name__)

class DocumentRetriever:
    """Class for retrieving relevant document snippets."""
    
    def __init__(self, embedding_model="all-MiniLM-L6-v2", chunk_size=500, chunk_overlap=100, cache_dir="cache"):
        """Initialize the document retriever.
        
        Args:
            embedding_model: Name of the embedding model to use
            chunk_size: Size of chunks in characters
            chunk_overlap: Overlap between chunks in characters
            cache_dir: Directory to cache embeddings
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.cache_dir = cache_dir
        
        # Create components
        self.chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = TextEmbedder(model_name=embedding_model, cache_dir=cache_dir)
        self.ranker = SnippetRanker()
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk documents into smaller pieces.
        
        Args:
            documents: List of document dictionaries with 'content' field
            
        Returns:
            list: List of chunk dictionaries
        """
        all_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            # Skip documents without content
            if not doc.get('content'):
                continue
                
            # Ensure document has an ID
            doc_id = doc.get('id', f"doc_{doc_idx}")
            
            # Add document metadata to each chunk
            doc_metadata = doc.get('metadata', {})
            if 'title' in doc:
                doc_metadata['title'] = doc['title']
            if 'url' in doc:
                doc_metadata['url'] = doc['url']
            
            # Chunk the document
            doc_chunks = self.chunker.chunk_document({
                'content': doc['content'],
                'id': doc_id,
                'metadata': doc_metadata
            })
            
            all_chunks.extend(doc_chunks)
        
        return all_chunks
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Chunk a plain text into smaller pieces.
        
        Args:
            text: Text content
            metadata: Optional metadata to include with each chunk
            
        Returns:
            list: List of chunk dictionaries
        """
        return self.chunker.chunk_text(text, metadata)
    
    def retrieve_relevant_chunks(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve the most relevant chunks for a query.
        
        Args:
            query: Query string
            chunks: List of chunk dictionaries
            top_k: Number of chunks to retrieve
            
        Returns:
            list: List of the most relevant chunks
        """
        if not query or not chunks:
            return []
            
        # Get query embedding
        query_embedding = self.embedder.get_embedding(query)
        
        # Get chunk embeddings
        chunk_contents = [chunk.get('content', '') for chunk in chunks]
        chunk_embeddings = self.embedder.get_embeddings(chunk_contents)
        
        # Calculate relevance scores
        scores = []
        for i, embedding in enumerate(chunk_embeddings):
            # Skip empty chunks
            if np.all(embedding == 0):
                scores.append(0.0)
                continue
                
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            scores.append(float(similarity))
        
        # Get top-k chunks by score
        top_indices = np.argsort(scores)[-top_k:][::-1]  # Descending order
        
        # Return top chunks with their scores
        return [
            {**chunks[i], 'relevance_score': scores[i]}
            for i in top_indices
            if scores[i] > 0  # Only include chunks with positive scores
        ]
    
    def rerank_chunks(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Rerank chunks based on relevance to query.
        
        Args:
            query: Query string
            chunks: List of chunk dictionaries with 'relevance_score'
            top_k: Number of chunks to return after reranking
            
        Returns:
            list: List of reranked chunks
        """
        # Skip if no chunks or query
        if not query or not chunks:
            return []
            
        # Rerank chunks
        reranked_chunks = self.ranker.rerank(query, chunks)
        
        # Get top-k reranked chunks
        return reranked_chunks[:top_k]
    
    def chunk_and_retrieve(self, query: str, texts: List[str], top_k: int = 5) -> List[str]:
        """Chunk texts and retrieve relevant chunks for a query.
        
        Args:
            query: Query string
            texts: List of text strings
            top_k: Number of chunks to retrieve
            
        Returns:
            list: List of relevant text chunks
        """
        # Skip if no texts or query
        if not query or not texts:
            return []
            
        # Create documents from texts
        documents = [{'content': text, 'id': f"doc_{i}"} for i, text in enumerate(texts)]
        
        # Chunk documents
        chunks = self.chunk_documents(documents)
        
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(query, chunks, top_k)
        
        # Return the content of relevant chunks
        return [chunk.get('content', '') for chunk in relevant_chunks]
    
    def search(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents and chunks.
        
        Args:
            query: Query string
            documents: List of document dictionaries
            top_k: Number of chunks to retrieve
            
        Returns:
            list: List of relevant document chunks with metadata
        """
        # Skip if no documents or query
        if not query or not documents:
            return []
            
        # Chunk documents
        chunks = self.chunk_documents(documents)
        
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(query, chunks, top_k * 2)
        
        # Rerank chunks to improve quality
        reranked_chunks = self.rerank_chunks(query, relevant_chunks, top_k)
        
        return reranked_chunks