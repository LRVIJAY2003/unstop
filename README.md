"""
Text embedding utilities for the RAG system.
"""
import logging
import os
import numpy as np
import pickle
from typing import List, Dict, Any, Union

logger = logging.getLogger(__name__)

class TextEmbedder:
    """Class for embedding text using various models."""
    
    def __init__(self, model_name="all-MiniLM-L6-v2", cache_dir="cache"):
        """Initialize the text embedder.
        
        Args:
            model_name: Name of the sentence transformer model
            cache_dir: Directory to cache embeddings
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Embedding cache
        self.embedding_cache = {}
        self.embedding_cache_file = os.path.join(cache_dir, f"{model_name.replace('/', '_')}_cache.pkl")
        
        # Load cache if it exists
        self._load_cache()
    
    def load_model(self):
        """Load the sentence transformer model."""
        if self.model is not None:
            return
            
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded sentence transformer model: {self.model_name}")
        except ImportError:
            logger.warning("SentenceTransformer not available, falling back to simpler methods")
            self.model = None
    
    def _load_cache(self):
        """Load embedding cache from disk."""
        if os.path.exists(self.embedding_cache_file):
            try:
                with open(self.embedding_cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
            except Exception as e:
                logger.error(f"Error loading embedding cache: {str(e)}")
                self.embedding_cache = {}
    
    def _save_cache(self):
        """Save embedding cache to disk."""
        try:
            with open(self.embedding_cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            logger.info(f"Saved {len(self.embedding_cache)} embeddings to cache")
        except Exception as e:
            logger.error(f"Error saving embedding cache: {str(e)}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            np.ndarray: Embedding vector
        """
        if not text:
            # Return zero vector for empty text
            return np.zeros(384)  # Default size for most models
            
        # Check cache
        if text in self.embedding_cache:
            return self.embedding_cache[text]
            
        # Load model if not already loaded
        self.load_model()
        
        if self.model:
            # Generate embedding using sentence transformer
            embedding = self.model.encode(text, convert_to_numpy=True)
        else:
            # Fall back to simpler method if model is not available
            embedding = self._simple_embedding(text)
            
        # Cache the embedding
        self.embedding_cache[text] = embedding
        
        # Save cache periodically (every 100 new embeddings)
        if len(self.embedding_cache) % 100 == 0:
            self._save_cache()
            
        return embedding
    
    def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            list: List of embedding vectors
        """
        # Filter out empty texts
        non_empty_texts = [(i, text) for i, text in enumerate(texts) if text]
        indices, valid_texts = zip(*non_empty_texts) if non_empty_texts else ([], [])
        
        # Check cache for each text
        embeddings = [None] * len(texts)
        texts_to_embed = []
        texts_indices = []
        
        for i, text in zip(indices, valid_texts):
            if text in self.embedding_cache:
                embeddings[i] = self.embedding_cache[text]
            else:
                texts_to_embed.append(text)
                texts_indices.append(i)
        
        if texts_to_embed:
            # Load model if not already loaded
            self.load_model()
            
            if self.model:
                # Generate embeddings using sentence transformer
                batch_embeddings = self.model.encode(texts_to_embed, convert_to_numpy=True)
            else:
                # Fall back to simpler method
                batch_embeddings = [self._simple_embedding(text) for text in texts_to_embed]
                
            # Update embeddings and cache
            for i, embedding in zip(texts_indices, batch_embeddings):
                embeddings[i] = embedding
                self.embedding_cache[texts[i]] = embedding
                
            # Save cache if we've added several new embeddings
            if len(texts_to_embed) > 10:
                self._save_cache()
        
        # Fill in empty slots with zero vectors
        default_dim = 384  # Default size
        if any(embeddings):
            default_dim = embeddings[indices[0]].shape[0] if indices else default_dim
            
        for i in range(len(embeddings)):
            if embeddings[i] is None:
                embeddings[i] = np.zeros(default_dim)
                
        return embeddings
    
    def _simple_embedding(self, text: str) -> np.ndarray:
        """Simple fallback embedding method.
        
        Args:
            text: Text to embed
            
        Returns:
            np.ndarray: Embedding vector
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Create a small corpus with just this text
            corpus = [text]
            vectorizer = TfidfVectorizer(max_features=384)
            tfidf_matrix = vectorizer.fit_transform(corpus)
            
            # Convert to dense and resize to standard dimensions
            embedding = tfidf_matrix.toarray()[0]
            
            # Pad or truncate to get standard size
            if len(embedding) < 384:
                embedding = np.pad(embedding, (0, 384 - len(embedding)))
            elif len(embedding) > 384:
                embedding = embedding[:384]
                
            return embedding
            
        except ImportError:
            # If sklearn is not available, use an even simpler approach
            logger.warning("TfidfVectorizer not available, using very simple embedding")
            
            # Create a simple bag-of-words vector
            words = set(text.lower().split())
            # Simple hash-based embedding
            embedding = np.zeros(384)
            
            for word in words:
                # Simple hash function to distribute words
                hash_val = hash(word) % 384
                embedding[hash_val] += 1
                
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            return embedding
    
    def get_document_embeddings(self, documents: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Get embeddings for a list of documents.
        
        Args:
            documents: List of document dictionaries with 'content' field
            
        Returns:
            dict: Dictionary mapping document IDs to embedding vectors
        """
        # Extract document contents and IDs
        contents = [doc.get('content', '') for doc in documents]
        doc_ids = [str(i) for i in range(len(documents))]
        
        # Get embeddings
        embeddings = self.get_embeddings(contents)
        
        # Create dictionary mapping document IDs to embeddings
        return {doc_id: embedding for doc_id, embedding in zip(doc_ids, embeddings)}
    
    def get_chunk_embeddings(self, chunks: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Get embeddings for a list of chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'content' and 'chunk_id' fields
            
        Returns:
            dict: Dictionary mapping chunk IDs to embedding vectors
        """
        # Extract chunk contents and IDs
        contents = [chunk.get('content', '') for chunk in chunks]
        chunk_ids = [f"{i}" for i in range(len(chunks))]
        
        # Get embeddings
        embeddings = self.get_embeddings(contents)
        
        # Create dictionary mapping chunk IDs to embeddings
        return {chunk_id: embedding for chunk_id, embedding in zip(chunk_ids, embeddings)}