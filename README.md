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