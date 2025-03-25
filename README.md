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