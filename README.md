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