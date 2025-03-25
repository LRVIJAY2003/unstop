"""
Document chunking utilities for the RAG system.
"""
import logging
import re
import nltk
from typing import List, Dict, Any

# Download necessary NLTK data
try:
    nltk.download('punkt', quiet=True)
except:
    pass

logger = logging.getLogger(__name__)

class DocumentChunker:
    """Class for chunking documents into manageable pieces."""
    
    def __init__(self, chunk_size=500, chunk_overlap=100):
        """Initialize the document chunker.
        
        Args:
            chunk_size: Target size of chunks (in characters)
            chunk_overlap: Overlap between chunks (in characters)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk a document into smaller pieces.
        
        Args:
            document: Document as a dictionary with 'content' field
            
        Returns:
            list: List of chunk dictionaries
        """
        if not document or 'content' not in document:
            return []
        
        content = document['content']
        
        # If the content is already small enough, return as a single chunk
        if len(content) <= self.chunk_size:
            return [{
                **document,
                'chunk_id': 0,
                'content': content,
                'chunk_type': 'full_document'
            }]
        
        # First, try to split by section headings
        heading_pattern = r'(?:\n|^)(#{1,3} .+)(?:\n|$)'
        headings = list(re.finditer(heading_pattern, content))
        
        # If we have headings, use them as chunk boundaries
        if len(headings) > 1:
            return self._chunk_by_headings(document, content, headings)
        
        # Otherwise, fall back to paragraph chunking
        return self._chunk_by_paragraphs(document, content)
    
    def _chunk_by_headings(self, document: Dict[str, Any], content: str, headings: List[re.Match]) -> List[Dict[str, Any]]:
        """Chunk content using headings as boundaries.
        
        Args:
            document: Original document
            content: Document content
            headings: List of heading regex matches
            
        Returns:
            list: List of chunk dictionaries
        """
        chunks = []
        
        # Extract title if first heading is at the start of the document
        title = None
        if headings[0].start() == 0:
            title_end = headings[1].start() if len(headings) > 1 else len(content)
            title = content[:title_end].strip()
            
            # Add title as a separate chunk
            chunks.append({
                **document,
                'chunk_id': 0,
                'content': title,
                'chunk_type': 'title'
            })
        
        # Process remaining content by heading
        for i in range(len(headings)):
            start = headings[i].start()
            end = headings[i+1].start() if i < len(headings) - 1 else len(content)
            
            section = content[start:end].strip()
            
            # Skip if this is the title we already added
            if section == title:
                continue
            
            # If section is too large, subdivide further
            if len(section) > self.chunk_size:
                paragraphs = re.split(r'\n\n+', section)
                current_chunk = []
                current_size = 0
                
                for para in paragraphs:
                    if current_size + len(para) > self.chunk_size and current_chunk:
                        # Save current chunk
                        combined = '\n\n'.join(current_chunk)
                        chunks.append({
                            **document,
                            'chunk_id': len(chunks),
                            'content': combined,
                            'chunk_type': 'section_part'
                        })
                        
                        # Start new chunk with overlap
                        overlap_point = max(0, len(current_chunk) - 1)
                        current_chunk = current_chunk[overlap_point:]
                        current_size = sum(len(p) for p in current_chunk)
                    
                    current_chunk.append(para)
                    current_size += len(para)
                
                # Add the last chunk if it exists
                if current_chunk:
                    combined = '\n\n'.join(current_chunk)
                    chunks.append({
                        **document,
                        'chunk_id': len(chunks),
                        'content': combined,
                        'chunk_type': 'section_part'
                    })
            else:
                # Add section as a single chunk
                chunks.append({
                    **document,
                    'chunk_id': len(chunks),
                    'content': section,
                    'chunk_type': 'section'
                })
        
        return chunks
    
    def _chunk_by_paragraphs(self, document: Dict[str, Any], content: str) -> List[Dict[str, Any]]:
        """Chunk content by paragraphs.
        
        Args:
            document: Original document
            content: Document content
            
        Returns:
            list: List of chunk dictionaries
        """
        chunks = []
        paragraphs = re.split(r'\n\n+', content)
        
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            # If adding this paragraph would exceed chunk size, save current chunk and start a new one
            if current_size + len(para) > self.chunk_size and current_chunk:
                combined = '\n\n'.join(current_chunk)
                chunks.append({
                    **document,
                    'chunk_id': len(chunks),
                    'content': combined,
                    'chunk_type': 'paragraph_group'
                })
                
                # Start new chunk with overlap
                overlap_point = max(0, len(current_chunk) - 2)  # Include up to 2 paragraphs overlap
                current_chunk = current_chunk[overlap_point:]
                current_size = sum(len(p) for p in current_chunk)
            
            # Add paragraph to current chunk
            current_chunk.append(para)
            current_size += len(para)
        
        # Add the last chunk if it exists
        if current_chunk:
            combined = '\n\n'.join(current_chunk)
            chunks.append({
                **document,
                'chunk_id': len(chunks),
                'content': combined,
                'chunk_type': 'paragraph_group'
            })
        
        return chunks
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Chunk a plain text into smaller pieces.
        
        Args:
            text: Text content
            metadata: Optional metadata to include with each chunk
            
        Returns:
            list: List of chunk dictionaries
        """
        document = {
            'content': text,
            'metadata': metadata or {}
        }
        
        return self.chunk_document(document)
    
    def chunk_by_sentences(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Chunk text by sentences, respecting chunk size.
        
        Args:
            text: Text content
            metadata: Optional metadata to include with each chunk
            
        Returns:
            list: List of chunk dictionaries
        """
        if not text:
            return []
            
        # Get all sentences
        sentences = nltk.sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # Handle very large sentences
            if sentence_size > self.chunk_size:
                # If we have content in the current chunk, add it first
                if current_chunk:
                    chunks.append({
                        'content': ' '.join(current_chunk),
                        'chunk_id': len(chunks),
                        'chunk_type': 'sentence_group',
                        'metadata': metadata or {}
                    })
                    current_chunk = []
                    current_size = 0
                
                # Split the long sentence
                words = sentence.split()
                current_sentence_chunk = []
                current_sentence_size = 0
                
                for word in words:
                    if current_sentence_size + len(word) + 1 > self.chunk_size and current_sentence_chunk:
                        chunks.append({
                            'content': ' '.join(current_sentence_chunk),
                            'chunk_id': len(chunks),
                            'chunk_type': 'sentence_part',
                            'metadata': metadata or {}
                        })
                        current_sentence_chunk = []
                        current_sentence_size = 0
                    
                    current_sentence_chunk.append(word)
                    current_sentence_size += len(word) + 1  # Include space
                
                # Add any remaining words
                if current_sentence_chunk:
                    chunks.append({
                        'content': ' '.join(current_sentence_chunk),
                        'chunk_id': len(chunks),
                        'chunk_type': 'sentence_part',
                        'metadata': metadata or {}
                    })
            
            # For normal sentences, add to current chunk
            elif current_size + sentence_size > self.chunk_size and current_chunk:
                # Save current chunk and start a new one
                chunks.append({
                    'content': ' '.join(current_chunk),
                    'chunk_id': len(chunks),
                    'chunk_type': 'sentence_group',
                    'metadata': metadata or {}
                })
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append({
                'content': ' '.join(current_chunk),
                'chunk_id': len(chunks),
                'chunk_type': 'sentence_group',
                'metadata': metadata or {}
            })
        
        return chunks













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












"""
Re-ranking utilities for the RAG system.
"""
import logging
import re
import numpy as np
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class SnippetRanker:
    """Class for ranking and re-ranking document snippets."""
    
    def __init__(self):
        """Initialize the snippet ranker."""
        pass
    
    def rerank(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank chunks based on relevance to query.
        
        Args:
            query: Query string
            chunks: List of chunk dictionaries with 'relevance_score'
            
        Returns:
            list: List of reranked chunks
        """
        # Skip if no chunks or query
        if not query or not chunks:
            return []
            
        # Extract keywords from query
        query_keywords = self._extract_keywords(query)
        
        # Calculate enhanced scores
        enhanced_scores = []
        for chunk in chunks:
            # Start with base relevance score
            base_score = chunk.get('relevance_score', 0.0)
            
            # Apply various reranking factors
            keyword_bonus = self._calculate_keyword_bonus(query_keywords, chunk.get('content', ''))
            structural_bonus = self._calculate_structural_bonus(chunk)
            length_penalty = self._calculate_length_penalty(chunk.get('content', ''))
            
            # Calculate final score
            enhanced_score = base_score + keyword_bonus + structural_bonus - length_penalty
# Add enhanced score to chunk
            chunk['enhanced_score'] = enhanced_score
            enhanced_scores.append((chunk, enhanced_score))
        
        # Sort chunks by enhanced score (descending)
        reranked_chunks = [chunk for chunk, _ in sorted(enhanced_scores, key=lambda x: x[1], reverse=True)]
        
        return reranked_chunks
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            list: List of keywords
        """
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Split into words
        words = text.split()
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                     'be', 'been', 'being', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 
                     'about', 'as', 'of', 'that', 'this', 'these', 'those', 'it', 'its'}
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def _calculate_keyword_bonus(self, query_keywords: List[str], content: str) -> float:
        """Calculate bonus based on keyword presence.
        
        Args:
            query_keywords: List of keywords from query
            content: Content text
            
        Returns:
            float: Keyword bonus score
        """
        if not query_keywords or not content:
            return 0.0
            
        # Convert content to lowercase for case-insensitive matching
        content_lower = content.lower()
        
        # Count keyword matches
        matches = sum(1 for keyword in query_keywords if keyword in content_lower)
        
        # Calculate match ratio
        match_ratio = matches / len(query_keywords) if query_keywords else 0
        
        # Apply bonus (up to 0.3)
        return match_ratio * 0.3
    
    def _calculate_structural_bonus(self, chunk: Dict[str, Any]) -> float:
        """Calculate bonus based on chunk structure.
        
        Args:
            chunk: Chunk dictionary
            
        Returns:
            float: Structural bonus score
        """
        bonus = 0.0
        
        # Boost chunks with titles
        if chunk.get('chunk_type') == 'title':
            bonus += 0.2
        
        # Boost chunks with section headings
        elif chunk.get('chunk_type') == 'section':
            bonus += 0.15
        
        # Smaller boost for section parts
        elif chunk.get('chunk_type') == 'section_part':
            bonus += 0.1
        
        # Boost chunks from document beginnings
        if chunk.get('chunk_id', 0) == 0:
            bonus += 0.1
        
        return bonus
    
    def _calculate_length_penalty(self, content: str) -> float:
        """Calculate penalty based on content length.
        
        Args:
            content: Content text
            
        Returns:
            float: Length penalty score
        """
        if not content:
            return 0.0
            
        # Count words
        word_count = len(content.split())
        
        # Penalize very short chunks
        if word_count < 30:
            return 0.1
            
        # Penalize very long chunks
        if word_count > 300:
            return 0.05
            
        return 0.0










"""
Client for interacting with Google's Gemini API.
"""
import logging
import os
import time
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class GeminiClient:
    """Client for interacting with Google's Gemini API."""
    
    def __init__(self, project_id: str, model_name: str = "gemini-1.0-flash-001", region: str = "us-central1"):
        """Initialize the Gemini client.
        
        Args:
            project_id: Google Cloud project ID
            model_name: Gemini model name
            region: Google Cloud region
        """
        self.project_id = project_id
        self.model_name = model_name
        self.region = region
        
        # Initialize the client
        self.client = None
        self.model = None
        self.generation_config = None
        
        # Initialize Gemini
        self._initialize_gemini()
    
    def _initialize_gemini(self):
        """Initialize the Gemini client."""
        try:
            import google.generativeai as genai
            
            self.client = genai.Client(
                project_id=self.project_id,
                location=self.region
            )
            
            # Get the model
            self.model = self.model_name
            
            # Set up generation config
            self.generation_config = {
                "temperature": 0.5,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 4096,
                "response_mime_type": "text/plain",
            }
            
            logger.info(f"Initialized Gemini client with model {self.model}")
            
        except ImportError:
            logger.error("Failed to import Google Generative AI package")
            raise ImportError("google-generativeai package is required")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {str(e)}")
            raise
    
    def check_connection(self):
        """Check if the connection to Gemini is working.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            # Simple test query
            response = self.generate_response_from_prompt("Hello, are you working?")
            return response is not None
        except Exception as e:
            logger.error(f"Failed to connect to Gemini: {str(e)}")
            return False
    
    def generate_response_from_prompt(self, prompt, instructions=None):
        """
        Uses Vertex AI to generate response from a prompt.
        
        Args:
            prompt (str): The prompt.
            instructions (str, optional): Optional system instructions.
            
        Returns:
            str: The response.
        """
        try:
            import google.generativeai as genai
            
            logger.info(f"Project ID: {self.project_id}")
            logger.info(f"Location: {self.region}")
            logger.info(f"Model: {self.model_name}")
            logger.debug(f"Prompt: {prompt}")
            
            # Prepare safety settings - all set to "OFF" for this demo
            # In a production environment, you should adjust these settings appropriately
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                }
            ]
            
            # Create content
            content = [prompt]
            
            # Add system instructions if provided
            if instructions:
                generation_config = {
                    **self.generation_config,
                    "system_instruction": instructions
                }
            else:
                generation_config = self.generation_config
            
            # Stream the response
            responses = self.client.generate_content(
                model=self.model,
                contents=content,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            response_text = ""
            for response in responses:
                for part in response.parts:
                    response_text += part.text
            
            return response_text
            
        except Exception as e:
            logger.error(f"[{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}] Failed to run the program with exception: {e}")
            raise e
    
    def generate_structured_output(self, prompt, output_schema, system_instruction=None):
        """Generate structured output based on a schema.
        
        Args:
            prompt: Input prompt
            output_schema: JSON schema for the output
            system_instruction: Optional system instruction
            
        Returns:
            dict: Structured output
        """
        try:
            import google.generativeai as genai
            
            # Prepare the prompt
            full_prompt = f"""
            {prompt}
            
            Please format your response as a JSON object matching this schema:
            {json.dumps(output_schema, indent=2)}
            
            IMPORTANT: Your response should be valid JSON only.
            """
            
            # Generate response
            response_text = self.generate_response_from_prompt(full_prompt, system_instruction)
            
            # Extract JSON from response
            json_pattern = r'```json\s+(.*?)\s+```|({.*})'
            json_match = re.search(json_pattern, response_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1) or json_match.group(2)
                return json.loads(json_str)
            else:
                try:
                    # Try parsing the whole response as JSON
                    return json.loads(response_text)
                except json.JSONDecodeError:
                    logger.error("Failed to extract JSON from response")
                    return None
                
        except Exception as e:
            logger.error(f"Error generating structured output: {str(e)}")
            return None










"""
Logging utilities for the application.
"""
import logging
import os
import sys
from datetime import datetime

def setup_logger(log_level="INFO", log_file=None):
    """Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        
    Returns:
        logging.Logger: Configured logger
    """
    # Convert log level string to logging level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add console handler to logger
    logger.addHandler(console_handler)
    
    # Add file handler if log file is specified
    if log_file:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        
        # Add file handler to logger
        logger.addHandler(file_handler)
    
    return logger









"""
Caching utilities for the application.
"""
import os
import time
import json
import pickle
import hashlib
import logging
from functools import wraps
from typing import Any, Dict, Callable

logger = logging.getLogger(__name__)

def get_cache_key(func_name, args, kwargs):
    """Generate a cache key from function name and arguments.
    
    Args:
        func_name: Function name
        args: Function positional arguments
        kwargs: Function keyword arguments
        
    Returns:
        str: Cache key
    """
    # Convert arguments to string representation
    arg_str = str(args) + str(sorted(kwargs.items()))
    
    # Hash the argument string
    return f"{func_name}_{hashlib.md5(arg_str.encode()).hexdigest()}"

def cache_result(ttl=3600, cache_dir="cache"):
    """Decorator to cache function results.
    
    Args:
        ttl: Time-to-live in seconds (default: 1 hour)
        cache_dir: Directory to store cache files
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache directory if it doesn't exist
            os.makedirs(cache_dir, exist_ok=True)
            
            # Generate cache key
            cache_key = get_cache_key(func.__name__, args, kwargs)
            cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
            
            # Check if cache file exists and is not expired
            if os.path.exists(cache_file):
                # Get file modification time
                mod_time = os.path.getmtime(cache_file)
                
                # Check if cache is still valid
                if time.time() - mod_time < ttl:
                    try:
                        with open(cache_file, 'rb') as f:
                            result = pickle.load(f)
                        logger.debug(f"Cache hit for {func.__name__}")
                        return result
                    except Exception as e:
                        logger.warning(f"Error loading cache for {func.__name__}: {str(e)}")
            
            # Cache miss or error, call the function
            result = func(*args, **kwargs)
            
            # Save result to cache
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                logger.debug(f"Cached result for {func.__name__}")
            except Exception as e:
                logger.warning(f"Error caching result for {func.__name__}: {str(e)}")
            
            return result
        
        return wrapper
    
    return decorator

class SimpleCache:
    """Simple in-memory cache with expiration."""
    
    def __init__(self, ttl=3600):
        """Initialize the cache.
        
        Args:
            ttl: Time-to-live in seconds (default: 1 hour)
        """
        self.cache = {}
        self.ttl = ttl
    
    def get(self, key):
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Any: Cached value or None if not found
        """
        if key not in self.cache:
            return None
            
        value, timestamp = self.cache[key]
        
        # Check if value is expired
        if time.time() - timestamp > self.ttl:
            # Remove expired value
            del self.cache[key]
            return None
            
        return value
    
    def set(self, key, value):
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        self.cache[key] = (value, time.time())
    
    def delete(self, key):
        """Delete value from cache.
        
        Args:
            key: Cache key
        """
        if key in self.cache:
            del self.cache[key]
    
    def clear(self):
        """Clear all values from cache."""
        self.cache.clear()
    
    def get_or_set(self, key, func):
        """Get value from cache or set it using function.
        
        Args:
            key: Cache key
            func: Function to call if cache miss
            
        Returns:
            Any: Cached or newly computed value
        """
        value = self.get(key)
        
        if value is None:
            # Cache miss, call function
            value = func()
            self.set(key, value)
            
        return value











"""
Text processing utilities for the application.
"""
import re
import nltk
import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime

# Download necessary NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Get stopwords
try:
    STOPWORDS = set(stopwords.words('english'))
except:
    STOPWORDS = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
                    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'])

def preprocess_text(text: str) -> str:
    """Preprocess text for analysis.
    
    Args:
        text: Input text
        
    Returns:
        str: Preprocessed text
    """
    if not text:
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove numbers
    text = re.sub(r'\d+', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_text(text: str) -> List[str]:
    """Tokenize text into words.
    
    Args:
        text: Input text
        
    Returns:
        list: List of tokens
    """
    if not text:
        return []
        
    # Preprocess text
    text = preprocess_text(text)
    
    # Tokenize
    tokens = text.split()
    
    return tokens

def remove_stopwords(tokens: List[str]) -> List[str]:
    """Remove stopwords from tokens.
    
    Args:
        tokens: List of tokens
        
    Returns:
        list: Filtered tokens
    """
    return [token for token in tokens if token not in STOPWORDS]

def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """Lemmatize tokens.
    
    Args:
        tokens: List of tokens
        
    Returns:
        list: Lemmatized tokens
    """
    return [lemmatizer.lemmatize(token) for token in tokens]

def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """Extract keywords from text.
    
    Args:
        text: Input text
        top_n: Number of keywords to extract
        
    Returns:
        list: List of keywords
    """
    if not text:
        return []
        
    # Tokenize
    tokens = tokenize_text(text)
    
    # Remove stopwords
    filtered_tokens = remove_stopwords(tokens)
    
    # Lemmatize
    lemmatized_tokens = lemmatize_tokens(filtered_tokens)
    
    # Count token frequencies
    token_counts = {}
    for token in lemmatized_tokens:
        if len(token) > 2:  # Only consider tokens with length > 2
            token_counts[token] = token_counts.get(token, 0) + 1
    
    # Sort by frequency (descending)
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Get top-n keywords
    keywords = [token for token, _ in sorted_tokens[:top_n]]
    
    return keywords

def extract_named_entities(text: str) -> List[Tuple[str, str]]:
    """Extract named entities from text.
    
    Args:
        text: Input text
        
    Returns:
        list: List of (entity, entity_type) tuples
    """
    if not text:
        return []
        
    # Simple pattern-based entity extraction (as fallback)
    entities = []
    
    # Extract dates
    date_patterns = [
        r'\d{1,2}/\d{1,2}/\d{2,4}',  # MM/DD/YYYY or DD/MM/YYYY
        r'\d{1,2}-\d{1,2}-\d{2,4}',  # MM-DD-YYYY or DD-MM-YYYY
        r'\d{4}-\d{2}-\d{2}'         # YYYY-MM-DD (ISO format)
    ]
    
    for pattern in date_patterns:
        for match in re.finditer(pattern, text):
            entities.append((match.group(0), 'DATE'))
    
    # Extract emails
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    for match in re.finditer(email_pattern, text):
        entities.append((match.group(0), 'EMAIL'))
    
    # Extract URLs
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    for match in re.finditer(url_pattern, text):
        entities.append((match.group(0), 'URL'))
    
    return entities

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences.
    
    Args:
        text: Input text
        
    Returns:
        list: List of sentences
    """
    if not text:
        return []
        
    try:
        return nltk.sent_tokenize(text)
    except:
        # Fallback to simple sentence splitting
        return re.split(r'(?<=[.!?])\s+', text)

def clean_text(text: str) -> str:
    """Clean text by removing special characters and normalizing whitespace.
    
    Args:
        text: Input text
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
        
    # Replace special characters
    text = re.sub(r'[\r\n\t]+', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text













"""
HTML parsing utilities for the application.
"""
import re
import logging
from typing import List, Dict, Any
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

def extract_text_from_html(html: str) -> str:
    """Extract plain text from HTML.
    
    Args:
        html: HTML content
        
    Returns:
        str: Plain text
    """
    if not html:
        return ""
        
    try:
        # Parse HTML
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get text
        text = soup.get_text(separator=" ")
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
        
    except Exception as e:
        logger.error(f"Error extracting text from HTML: {str(e)}")
        return html  # Return original HTML if extraction fails

def extract_links_from_html(html: str) -> List[Dict[str, str]]:
    """Extract links from HTML.
    
    Args:
        html: HTML content
        
    Returns:
        list: List of dictionaries with link information
    """
    if not html:
        return []
        
    try:
        # Parse HTML
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract links
        links = []
        for a in soup.find_all('a', href=True):
            text = a.get_text().strip()
            href = a['href']
            
            links.append({
                'text': text,
                'url': href
            })
            
        return links
        
    except Exception as e:
        logger.error(f"Error extracting links from HTML: {str(e)}")
        return []

def extract_images_from_html(html: str) -> List[Dict[str, str]]:
    """Extract images from HTML.
    
    Args:
        html: HTML content
        
    Returns:
        list: List of dictionaries with image information
    """
    if not html:
        return []
        
    try:
        # Parse HTML
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract images
        images = []
        for img in soup.find_all('img'):
            src = img.get('src', '')
            alt = img.get('alt', '')
            
            if src:
                images.append({
                    'src': src,
                    'alt': alt
                })
            
        return images
        
    except Exception as e:
        logger.error(f"Error extracting images from HTML: {str(e)}")
        return []

def extract_tables_from_html(html: str) -> List[List[List[str]]]:
    """Extract tables from HTML.
    
    Args:
        html: HTML content
        
    Returns:
        list: List of tables, where each table is a list of rows and each row is a list of cells
    """
    if not html:
        return []
        
    try:
        # Parse HTML
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract tables
        tables = []
        for table in soup.find_all('table'):
            table_data = []
            
            # Process rows
            for tr in table.find_all('tr'):
                row_data = []
                
                # Process cells
                for td in tr.find_all(['td', 'th']):
                    cell_text = td.get_text().strip()
                    row_data.append(cell_text)
                
                if row_data:
                    table_data.append(row_data)
            
            if table_data:
                tables.append(table_data)
            
        return tables
        
    except Exception as e:
        logger.error(f"Error extracting tables from HTML: {str(e)}")
        return []

def html_to_markdown(html: str) -> str:
    """Convert HTML to Markdown.
    
    Args:
        html: HTML content
        
    Returns:
        str: Markdown text
    """
    if not html:
        return ""
        
    try:
        import html2text
        
        # Create converter
        h2t = html2text.HTML2Text()
        h2t.ignore_links = False
        h2t.ignore_images = False
        h2t.ignore_tables = False
        
        # Convert HTML to Markdown
        markdown = h2t.handle(html)
        
        return markdown
        
    except ImportError:
        logger.warning("html2text not available, falling back to simple text extraction")
        return extract_text_from_html(html)
    except Exception as e:
        logger.error(f"Error converting HTML to Markdown: {str(e)}")
        return extract_text_from_html(html)

def clean_html(html: str) -> str:
    """Clean HTML by removing unnecessary elements and attributes.
    
    Args:
        html: HTML content
        
    Returns:
        str: Cleaned HTML
    """
    if not html:
        return ""
        
    try:
        # Parse HTML
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for tag in soup(["script", "style"]):
            tag.extract()
        
        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        # Return cleaned HTML
        return str(soup)
        
    except Exception as e:
        logger.error(f"Error cleaning HTML: {str(e)}")
        return html  # Return original HTML if cleaning fails











"""
Input validation utilities for the application.
"""
import re
import logging
from typing import Any, Dict, List, Union, Optional

logger = logging.getLogger(__name__)

def validate_query(query: str) -> bool:
    """Validate a query string.
    
    Args:
        query: Query string
        
    Returns:
        bool: True if query is valid, False otherwise
    """
    if not query:
        return False
        
    # Check if query is too short
    if len(query.strip()) < 3:
        return False
        
    # Check if query is too long
    if len(query) > 1000:
        return False
        
    return True

def validate_url(url: str) -> bool:
    """Validate a URL.
    
    Args:
        url: URL string
        
    Returns:
        bool: True if URL is valid, False otherwise
    """
    if not url:
        return False
        
    # Check if URL is well-formed
    url_pattern = r'^https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    if not re.match(url_pattern, url):
        return False
        
    return True

def validate_email(email: str) -> bool:
    """Validate an email address.
    
    Args:
        email: Email string
        
    Returns:
        bool: True if email is valid, False otherwise
    """
    if not email:
        return False
        
    # Check if email is well-formed
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
        return False
        
    return True

def validate_api_token(token: str) -> bool:
    """Validate an API token.
    
    Args:
        token: API token string
        
    Returns:
        bool: True if token is valid, False otherwise
    """
    if not token:
        return False
        
    # Check if token is too short
    if len(token) < 10:
        return False
        
    return True

def validate_request_data(data: Dict[str, Any], required_fields: List[str]) -> Dict[str, str]:
    """Validate request data.
    
    Args:
        data: Request data
        required_fields: List of required field names
        
    Returns:
        dict: Dictionary of validation errors (field name -> error message)
    """
    errors = {}
    
    if not data:
        return {'general': 'No data provided'}
        
    # Check required fields
    for field in required_fields:
        if field not in data or not data[field]:
            errors[field] = f'Field "{field}" is required'
    
    return errors

def sanitize_input(input_str: str) -> str:
    """Sanitize input string.
    
    Args:
        input_str: Input string
        
    Returns:
        str: Sanitized string
    """
    if not input_str:
        return ""
        
    # Remove potential script tags
    sanitized = re.sub(r'<script.*?>.*?</script>', '', input_str, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove potential event handlers
    sanitized = re.sub(r' on\w+=".*?"', ' ', sanitized, flags=re.IGNORECASE)
    
    # Remove potential javascript: URLs
    sanitized = re.sub(r'javascript:', 'blocked:', sanitized, flags=re.IGNORECASE)
    
    return sanitized












<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}RAG System{% endblock %}</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
    {% block extra_css %}{% endblock %}
</head>
<body>
    <header>
        <div class="container">
            <div class="logo">
                <h1>Knowledge Assistant</h1>
            </div>
            <nav>
                <ul>
                    <li><a href="{{ url_for('index') }}">Home</a></li>
                </ul>
            </nav>
        </div>
    </header>

    <main>
        <div class="container">
            {% block content %}{% endblock %}
        </div>
    </main>

    <footer>
        <div class="container">
            <p>&copy; {{ now.year }} Knowledge Assistant</p>
        </div>
    </footer>

    <script src="{{ url_for('static', filename='js/main.js') }}"></script>
    {% block extra_js %}{% endblock %}
</body>
</html>








{% extends 'base.html' %}

{% block title %}Knowledge Assistant{% endblock %}

{% block extra_css %}
<link rel="stylesheet" href="{{ url_for('static', filename='css/chat.css') }}">
{% endblock %}

{% block content %}
<div class="chat-container">
    <div class="sidebar">
        <div class="data-sources">
            <h3>Data Sources</h3>
            <div class="checkbox-group">
                <label class="checkbox-container">
                    <input type="checkbox" id="source-confluence" name="source" value="confluence" checked>
                    <span class="checkmark"></span>
                    Confluence
                </label>
                <label class="checkbox-container">
                    <input type="checkbox" id="source-remedy" name="source" value="remedy" checked>
                    <span class="checkmark"></span>
                    Remedy
                </label>
            </div>
        </div>
        <div class="chat-history">
            <h3>Chat History</h3>
            <ul id="history-list">
                <!-- Chat history will be populated dynamically -->
            </ul>
        </div>
    </div>
    
    <div class="chat-main">
        <div class="chat-messages" id="chat-messages">
            <div class="message system">
                <div class="message-content">
                    <p>Hello! I'm your Knowledge Assistant. I can search our Confluence pages and Remedy tickets to answer your questions. What would you like to know?</p>
                </div>
            </div>
            <!-- Messages will be added dynamically -->
        </div>
        
        <div class="chat-input">
            <form id="query-form">
                <div class="input-group">
                    <input type="text" id="query-input" placeholder="Ask a question..." autocomplete="off">
                    <button type="submit" id="send-button">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="send-icon"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
                    </button>
                </div>
            </form>
        </div>
    </div>
    
    <div class="info-panel">
        <div class="panel-header">
            <h3>Information</h3>
            <button id="close-panel">×</button>
        </div>
        <div class="panel-content">
            <div id="source-info">
                <!-- Source information will be displayed here -->
                <p>Select a message to see source information.</p>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block extra_js %}
<script src="{{ url_for('static', filename='js/chat.js') }}"></script>
{% endblock %}










/* Base Styles */
:root {
    --primary-color: #1a73e8;
    --primary-light: #62a3ff;
    --primary-dark: #004baf;
    --secondary-color: #7986cb;
    --background-color: #f3f7ff;
    --surface-color: #ffffff;
    --text-color: #202124;
    --text-secondary: #5f6368;
    --border-color: #e0e0e0;
    --success-color: #34a853;
    --error-color: #ea4335;
    --warning-color: #fbbc05;
    
    --spacing-xs: 4px;
    --spacing-sm: 8px;
    --spacing-md: 16px;
    --spacing-lg: 24px;
    --spacing-xl: 32px;
    
    --border-radius-sm: 4px;
    --border-radius-md: 8px;
    --border-radius-lg: 16px;
    
    --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.12), 0 1px 2px rgba(0, 0, 0, 0.24);
    --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08);
    --shadow-lg: 0 10px 20px rgba(0, 0, 0, 0.1), 0 3px 6px rgba(0, 0, 0, 0.05);
    
    --font-main: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
    --font-size-xs: 12px;
    --font-size-sm: 14px;
    --font-size-md: 16px;
    --font-size-lg: 18px;
    --font-size-xl: 24px;
    
    --transition-fast: 150ms ease;
    --transition-normal: 300ms ease;
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

html, body {
    font-family: var(--font-main);
    font-size: var(--font-size-md);
    line-height: 1.5;
    color: var(--text-color);
    background-color: var(--background-color);
    height: 100%;
    width: 100%;
}

body {
    display: flex;
    flex-direction: column;
}

.container {
    width: 100%;
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 var(--spacing-md);
}

/* Header */
header {
    background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
    color: white;
    padding: var(--spacing-md) 0;
    box-shadow: var(--shadow-md);
}

header .container {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.logo h1 {
    font-size: var(--font-size-xl);
    font-weight: 600;
    margin: 0;
}

nav ul {
    list-style: none;
    display: flex;
}

nav li {
    margin-left: var(--spacing-lg);
}

nav a {
    color: white;
    text-decoration: none;
    font-weight: 500;
    transition: opacity var(--transition-fast);
}

nav a:hover {
    opacity: 0.8;
}

/* Main Content */
main {
    flex: 1;
    padding: var(--spacing-xl) 0;
}

/* Footer */
footer {
    background-color: var(--primary-dark);
    color: white;
    padding: var(--spacing-lg) 0;
    text-align: center;
    font-size: var(--font-size-sm);
}

/* Buttons */
.btn {
    display: inline-block;
    background-color: var(--primary-color);
    color: white;
    padding: var(--spacing-sm) var(--spacing-md);
    border-radius: var(--border-radius-sm);
    font-weight: 500;
    text-decoration: none;
    cursor: pointer;
    border: none;
    transition: background-color var(--transition-fast);
}

.btn:hover {
    background-color: var(--primary-dark);
}

.btn-secondary {
    background-color: var(--secondary-color);
}

.btn-secondary:hover {
    background-color: #5c6bc0;
}

.btn-ghost {
    background-color: transparent;
    color: var(--primary-color);
    border: 1px solid var(--primary-color);
}

.btn-ghost:hover {
    background-color: rgba(26, 115, 232, 0.05);
}

/* Cards */
.card {
    background-color: var(--surface-color);
    border-radius: var(--border-radius-md);
    box-shadow: var(--shadow-sm);
    padding: var(--spacing-lg);
    margin-bottom: var(--spacing-lg);
}

.card-title {
    font-size: var(--font-size-lg);
    font-weight: 600;
    margin-bottom: var(--spacing-md);
}

/* Form elements */
input, textarea, select {
    width: 100%;
    padding: var(--spacing-sm) var(--spacing-md);
    border: 1px solid var(--border-color);
    border-radius: var(--border-radius-sm);
    font-family: var(--font-main);
    font-size: var(--font-size-md);
    transition: border-color var(--transition-fast);
}

input:focus, textarea:focus, select:focus {
    outline: none;
    border-color: var(--primary-color);
}

label {
    display: block;
    margin-bottom: var(--spacing-xs);
    font-weight: 500;
}

/* Responsive */
@media (max-width: 768px) {
    .container {
        padding: 0 var(--spacing-md);
    }
    
    .logo h1 {
        font-size: var(--font-size-lg);
    }
    
    nav li {
        margin-left: var(--spacing-md);
    }
}











/* Chat-specific styles */
.chat-container {
    display: grid;
    grid-template-columns: 250px 1fr 300px;
    grid-gap: var(--spacing-md);
    height: calc(100vh - 180px);
    max-height: 800px;
    min-height: 500px;
}

/* Sidebar */
.sidebar {
    background-color: var(--surface-color);
    border-radius: var(--border-radius-md);
    box-shadow: var(--shadow-sm);
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.data-sources {
    padding: var(--spacing-md);
    border-bottom: 1px solid var(--border-color);
}

.data-sources h3 {
    margin-bottom: var(--spacing-md);
    font-size: var(--font-size-md);
    color: var(--text-secondary);
}

.checkbox-group {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-sm);
}

.checkbox-container {
    display: flex;
    align-items: center;
    cursor: pointer;
    user-select: none;
}

.checkbox-container input {
    position: absolute;
    opacity: 0;
    cursor: pointer;
    height: 0;
    width: 0;
}

.checkmark {
    position: relative;
    display: inline-block;
    height: 18px;
    width: 18px;
    margin-right: var(--spacing-sm);
    background-color: #fff;
    border: 2px solid var(--primary-color);
    border-radius: 3px;
}

.checkbox-container:hover input ~ .checkmark {
    background-color: #f0f7ff;
}

.checkbox-container input:checked ~ .checkmark {
    background-color: var(--primary-color);
}

.checkmark:after {
    content: "";
    position: absolute;
    display: none;
}

.checkbox-container input:checked ~ .checkmark:after {
    display: block;
}

.checkbox-container .checkmark:after {
    left: 5px;
    top: 1px;
    width: 4px;
    height: 10px;
    border: solid white;
    border-width: 0 2px 2px 0;
    transform: rotate(45deg);
}

.chat-history {
    flex: 1;
    padding: var(--spacing-md);
    overflow-y: auto;
}

.chat-history h3 {
    margin-bottom: var(--spacing-md);
    font-size: var(--font-size-md);
    color: var(--text-secondary);
}

#history-list {
    list-style: none;
}

#history-list li {
    padding: var(--spacing-sm) 0;
    border-bottom: 1px solid var(--border-color);
    font-size: var(--font-size-sm);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    cursor: pointer;
    transition: background-color var(--transition-fast);
}

#history-list li:hover {
    background-color: #f0f7ff;
}

/* Main Chat */
.chat-main {
    display: flex;
    flex-direction: column;
    background-color: var(--surface-color);
    border-radius: var(--border-radius-md);
    box-shadow: var(--shadow-sm);
    overflow: hidden;
}

.chat-messages {
    flex: 1;
    padding: var(--spacing-md);
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: var(--spacing-md);
}

.message {
    display: flex;
    align-items: flex-start;
    max-width: 80%;
}

.message.user {
    align-self: flex-end;
}

.message .avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    margin-right: var(--spacing-sm);
    background-color: var(--primary-light);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: bold;
}

.message.user .avatar {
    background-color: var(--secondary-color);
}

.message .message-content {
    background-color: #f0f7ff;
    padding: var(--spacing-md);
    border-radius: var(--border-radius-md);
    position: relative;
}

.message.user .message-content {
    background-color: var(--primary-color);
    color: white;
}

.message.system .message-content {
    background-color: #f5f5f5;
    color: var(--text-secondary);
}

.message .message-content:after {
    content: '';
    position: absolute;
    left: -8px;
    top: 10px;
    border-top: 8px solid transparent;
    border-bottom: 8px solid transparent;
    border-right: 8px solid #f0f7ff;
}

.message.user .message-content:after {
    left: auto;
    right: -8px;
    border-right: none;
    border-left: 8px solid var(--primary-color);
}

.message .message-content p {
    margin-bottom: var(--spacing-sm);
}

.message .message-content p:last-child {
    margin-bottom: 0;
}

.message .message-content ul,
.message .message-content ol {
    margin-left: var(--spacing-lg);
    margin-bottom: var(--spacing-sm);
}

.message .message-content code {
    font-family: monospace;
    background-color: rgba(0, 0, 0, 0.05);
    padding: 2px 4px;
    border-radius: 3px;
}

.message.user .message-content code {
    background-color: rgba(255, 255, 255, 0.2);
}

.message .message-content pre {
    font-family: monospace;
    background-color: rgba(0, 0, 0, 0.05);
    padding: var(--spacing-sm);
    border-radius: var(--border-radius-sm);
    overflow-x: auto;
    margin-bottom: var(--spacing-sm);
}

.message.user .message-content pre {
    background-color: rgba(255, 255, 255, 0.2);
}

.message .message-meta {
    font-size: var(--font-size-xs);
    color: var(--text-secondary);
    margin-top: var(--spacing-xs);
}

.message.user .message-meta {
    color: rgba(255, 255, 255, 0.8);
}

.chat-input {
    padding: var(--spacing-md);
    border-top: 1px solid var(--border-color);
}

.chat-input .input-group {
    display: flex;
    align-items: center;
}

.chat-input input {
    flex: 1;
    padding: var(--spacing-md);
    border: 1px solid var(--border-color);
    border-radius: var(--border-radius-lg);
    transition: border-color var(--transition-fast), box-shadow var(--transition-fast);
}

.chat-input input:focus {
    border-color: var(--primary-color);
    box-shadow: 0 0 0 2px rgba(26, 115, 232, 0.2);
}

.chat-input button {
    background-color: var(--primary-color);
    color: white;
    border: none;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-left: var(--spacing-sm);
    cursor: pointer;
    transition: background-color var(--transition-fast);
}

.chat-input button:hover {
    background-color: var(--primary-dark);
}

.chat-input .send-icon {
    width: 20px;
    height: 20px;
}

/* Info Panel */
.info-panel {
    background-color: var(--surface-color);
    border-radius: var(--border-radius-md);
    box-shadow: var(--shadow-sm);
    overflow: hidden;
    display: flex;
    flex-direction: column;
}

.panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: var(--spacing-md);
    border-bottom: 1px solid var(--border-color);
}

.panel-header h3 {
    font-size: var(--font-size-md);
    color: var(--text-secondary);
    margin: 0;
}

.panel-header button {
    background: none;
    border: none;
    color: var(--text-secondary);
    font-size: 24px;
    cursor: pointer;
    line-height: 1;
}

.panel-content {
    flex: 1;
    padding: var(--spacing-md);
    overflow-y: auto;
}

.source-item {
    padding: var(--spacing-sm);
    margin-bottom: var(--spacing-sm);
    border: 1px solid var(--border-color);
    border-radius: var(--border-radius-sm);
    background-color: #fbfbfb;
}

.source-item h4 {
    font-size: var(--font-size-sm);
    margin-bottom: var(--spacing-xs);
    color: var(--primary-color);
}

.source-item p {
    font-size: var(--font-size-xs);
    color: var(--text-secondary);
}

.source-item .source-link {
    display: block;
    margin-top: var(--spacing-xs);
    font-size: var(--font-size-xs);
    color: var(--primary-color);
    text-decoration: none;
}

.source-item .source-link:hover {
    text-decoration: underline;
}

/* Loading indicator */
.loading {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: var(--spacing-md);
}

.loading-dots {
    display: flex;
}

.loading-dots span {
    width: 8px;
    height: 8px;
    margin: 0 4px;
    background-color: var(--text-secondary);
    border-radius: 50%;
    animation: dot-pulse 1.5s infinite ease-in-out;
}

.loading-dots span:nth-child(2) {
    animation-delay: 0.2s;
}

.loading-dots span:nth-child(3) {
    animation-delay: 0.4s;
}

@keyframes dot-pulse {
    0%, 60%, 100% {
        transform: scale(1);
        opacity: 0.6;
    }
    30% {
        transform: scale(1.5);
        opacity: 1;
    }
}

/* Responsive */
@media (max-width: 1024px) {
    .chat-container {
        grid-template-columns: 200px 1fr;
    }
    
    .info-panel {
        display: none;
    }
}

@media (max-width: 768px) {
    .chat-container {
        grid-template-columns: 1fr;
    }
    
    .sidebar {
        display: none;
    }
}