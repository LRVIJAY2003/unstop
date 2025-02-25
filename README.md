# unstop
includes algorithms that will be scaled as per the requirements


import os
import re
import glob
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union, Set
from pathlib import Path
import tempfile
import docx
import PyPDF2
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from datetime import datetime
import string
import textwrap

# Install and import sumy for summarization
try:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    from sumy.summarizers.lsa import LsaSummarizer
    from sumy.nlp.stemmers import Stemmer
    from sumy.utils import get_stop_words
except ImportError:
    print("Installing sumy...")
    os.system("pip install sumy")
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    from sumy.summarizers.lsa import LsaSummarizer
    from sumy.nlp.stemmers import Stemmer
    from sumy.utils import get_stop_words

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

# Load spaCy model (if not already loaded)
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # If model not installed, download it
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class DocumentProcessor:
    """Handles document processing and retrieval."""
    
    def __init__(self, knowledge_base_path: str):
        """Initialize the document processor."""
        self.knowledge_base_path = knowledge_base_path
        self.documents = {}  # Will store document content
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.85,
            min_df=2
        )
        self.doc_embeddings = {}
        
    def load_all_documents(self) -> Dict[str, str]:
        """Load all documents from the knowledge base."""
        all_files = glob.glob(os.path.join(self.knowledge_base_path, '*.*'))
        
        for file_path in all_files:
            try:
                file_name = os.path.basename(file_path)
                file_extension = os.path.splitext(file_path)[1].lower()
                
                if file_extension == '.txt':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self.documents[file_name] = f.read()
                        
                elif file_extension == '.docx':
                    doc = docx.Document(file_path)
                    content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                    self.documents[file_name] = content
                    
                elif file_extension == '.pdf':
                    with open(file_path, 'rb') as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        content = ''
                        for page_num in range(len(pdf_reader.pages)):
                            content += pdf_reader.pages[page_num].extract_text()
                        self.documents[file_name] = content
                
                else:
                    print(f"Unsupported file type: {file_extension} for file {file_name}")
                    
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
                
        print(f"Loaded {len(self.documents)} documents from knowledge base")
        return self.documents
    
    def create_document_embeddings(self):
        """Create TF-IDF embeddings for all documents."""
        if not self.documents:
            self.load_all_documents()
            
        docs_content = list(self.documents.values())
        doc_names = list(self.documents.keys())
        
        # Fit and transform to get document embeddings
        tfidf_matrix = self.vectorizer.fit_transform(docs_content)
        
        # Store embeddings with their document names
        for i, doc_name in enumerate(doc_names):
            self.doc_embeddings[doc_name] = tfidf_matrix[i]
    
    def search_documents(self, query: str, top_k: int = 3) -> List[Tuple[str, float, str]]:
        """Search for documents relevant to the query."""
        if not self.doc_embeddings:
            self.create_document_embeddings()
        
        # Transform query using the same vectorizer
        query_vector = self.vectorizer.transform([query.lower()])
        
        results = []
        
        # Calculate similarity between query and all documents
        for doc_name, doc_vector in self.doc_embeddings.items():
            similarity = cosine_similarity(query_vector, doc_vector)[0][0]
            results.append((doc_name, similarity, self.documents[doc_name]))
            
        # Sort by similarity (highest first) and return top_k results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def keyword_search(self, keyword: str) -> List[Tuple[str, str]]:
        """Search for keyword matches in documents."""
        if not self.documents:
            self.load_all_documents()
            
        results = []
        keyword_lower = keyword.lower()
        
        for doc_name, content in self.documents.items():
            if keyword_lower in content.lower():
                # Extract context around the keyword
                context = self.extract_context(content, keyword_lower)
                if context:
                    results.append((doc_name, context))
                    
        return results
    
    def extract_context(self, text: str, keyword: str, context_size: int = 500) -> str:
        """Extract context around a keyword in text."""
        text_lower = text.lower()
        keyword_lower = keyword.lower()
        
        # Find all occurrences of the keyword
        matches = [match.start() for match in re.finditer(keyword_lower, text_lower)]
        
        if not matches:
            return ""
            
        # Extract context around first occurrence
        start_pos = max(0, matches[0] - context_size)
        end_pos = min(len(text), matches[0] + len(keyword) + context_size)
        
        # Find sentence boundaries
        if start_pos > 0:
            sentence_start = text.rfind('.', 0, start_pos)
            if sentence_start != -1:
                start_pos = sentence_start + 1
        
        if end_pos < len(text):
            sentence_end = text.find('.', end_pos)
            if sentence_end != -1:
                end_pos = sentence_end + 1
        
        return text[start_pos:end_pos].strip()
    
    def extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text."""
        doc = nlp(text)
        
        # Extract noun phrases and entities
        key_terms = []
        
        # Get named entities
        for ent in doc.ents:
            if ent.label_ in ('ORG', 'PRODUCT', 'GPE', 'PERSON', 'WORK_OF_ART', 'EVENT'):
                key_terms.append(ent.text)
                
        # Get noun phrases
        for chunk in doc.noun_chunks:
            # Only include multi-word phrases or important single nouns
            if len(chunk.text.split()) > 1 or (chunk.root.pos_ == 'NOUN' and chunk.root.tag_ not in ('NN', 'NNS')):
                key_terms.append(chunk.text)
        
        # Remove duplicates and sort by length (longer terms first)
        key_terms = list(set(key_terms))
        key_terms.sort(key=lambda x: len(x), reverse=True)
        
        return key_terms[:7]  # Return top 7 terms


class SummaryGenerator:
    """Generates concise summaries from retrieved documents."""
    
    def __init__(self):
        """Initialize the summary generator."""
        self.language = "english"
        self.stemmer = Stemmer(self.language)
        self.stop_words = get_stop_words(self.language)
        
        # Initialize summarizers
        self.lexrank = LexRankSummarizer(self.stemmer)
        self.lexrank.stop_words = self.stop_words
        
        self.lsa = LsaSummarizer(self.stemmer)
        self.lsa.stop_words = self.stop_words
    
    def generate_summary(self, texts: List[str], query: str, max_sentences: int = 10) -> str:
        """Generate a concise summary from multiple texts."""
        # Combine texts with appropriate weighting
        combined_text = self._prepare_text(texts, query)
        
        # Generate summary using multiple methods
        summary_sentences = self._extract_summary_sentences(combined_text, query, max_sentences)
        
        # Format into a readable, coherent summary
        formatted_summary = self._format_summary(summary_sentences, query)
        
        return formatted_summary
    
    def _prepare_text(self, texts: List[str], query: str) -> str:
        """Prepare and combine texts for summarization."""
        # Process query to identify important terms
        query_doc = nlp(query.lower())
        query_terms = set()
        
        for token in query_doc:
            if not token.is_stop and token.is_alpha and len(token.text) > 2:
                query_terms.add(token.lemma_)
        
        # Weight paragraphs containing query terms higher
        weighted_texts = []
        
        for text in texts:
            paragraphs = text.split('\n\n')
            weighted_paragraphs = []
            
            for para in paragraphs:
                # Skip very short paragraphs
                if len(para.split()) < 10:
                    continue
                    
                # Check for query term presence
                para_doc = nlp(para.lower())
                para_terms = {token.lemma_ for token in para_doc if token.is_alpha and not token.is_stop}
                
                # Count matching terms
                matches = len(query_terms.intersection(para_terms))
                
                # Add paragraph with repetition based on relevance
                if matches > 0:
                    # Repeat important paragraphs to increase their weight in the summary
                    repetitions = min(matches, 3)  # Up to 3 repetitions
                    for _ in range(repetitions):
                        weighted_paragraphs.append(para)
                else:
                    weighted_paragraphs.append(para)
            
            weighted_texts.append('\n\n'.join(weighted_paragraphs))
        
        return '\n\n'.join(weighted_texts)
    
    def _extract_summary_sentences(self, text: str, query: str, max_sentences: int) -> List[str]:
        """Extract key sentences for the summary using multiple methods."""
        # Parse the text
        parser = PlaintextParser.from_string(text, Tokenizer(self.language))
        
        # Get sentences from different summarizers
        lexrank_sentences = [str(s) for s in self.lexrank(parser.document, max_sentences)]
        lsa_sentences = [str(s) for s in self.lsa(parser.document, max_sentences // 2)]
        
        # Also get query-focused sentences
        query_sentences = self._get_query_focused_sentences(text, query, max_sentences // 2)
        
        # Combine sentences from different methods
        all_sentences = []
        
        # First add query-focused sentences
        all_sentences.extend(query_sentences)
        
        # Then add LexRank sentences not already included
        for sentence in lexrank_sentences:
            if sentence not in all_sentences:
                all_sentences.append(sentence)
                
        # Then add LSA sentences not already included
        for sentence in lsa_sentences:
            if sentence not in all_sentences:
                all_sentences.append(sentence)
        
        # Limit to max_sentences
        return all_sentences[:max_sentences]
    
    def _get_query_focused_sentences(self, text: str, query: str, count: int) -> List[str]:
        """Extract sentences most relevant to the query."""
        # Get query terms
        query_doc = nlp(query.lower())
        query_terms = [token.lemma_ for token in query_doc 
                      if not token.is_stop and token.is_alpha and len(token.text) > 2]
        
        if not query_terms:
            return []
        
        # Extract sentences
        sentences = nltk.sent_tokenize(text)
        
        # Score sentences based on query relevance
        scored_sentences = []
        for sentence in sentences:
            if len(sentence.split()) < 5:  # Skip very short sentences
                continue
                
            # Process sentence
            sent_doc = nlp(sentence.lower())
            sent_terms = [token.lemma_ for token in sent_doc 
                         if token.is_alpha and not token.is_stop]
            
            # Count query term matches
            matches = sum(1 for term in query_terms if term in sent_terms)
            
            # Score as percentage of query terms matched
            score = matches / max(1, len(query_terms))
            
            if score > 0:  # Only consider sentences with at least one match
                scored_sentences.append((score, sentence))
        
        # Sort by score
        scored_sentences.sort(reverse=True)
        
        # Return top sentences
        return [s for _, s in scored_sentences[:count]]
    
    def _format_summary(self, sentences: List[str], query: str) -> str:
        """Format the summary sentences into a coherent text."""
        if not sentences:
            return "No relevant information found."
            
        # First, cluster related sentences
        clusters = self._cluster_related_sentences(sentences)
        
        # Start with a topic sentence
        intro_sentence = self._create_intro_sentence(sentences, query)
        
        # Format the summary
        formatted_paragraphs = [intro_sentence]
        
        for cluster in clusters:
            if len(cluster) == 1:
                formatted_paragraphs.append(cluster[0])
            else:
                # Combine cluster sentences into a paragraph
                formatted_paragraphs.append(' '.join(cluster))
        
        # Add a conclusion sentence
        conclusion = self._create_conclusion_sentence(sentences, query)
        if conclusion:
            formatted_paragraphs.append(conclusion)
            
        return '\n\n'.join(formatted_paragraphs)
    
    def _cluster_related_sentences(self, sentences: List[str]) -> List[List[str]]:
        """Group related sentences together."""
        if len(sentences) <= 1:
            return [sentences]
            
        # Process sentences to get their representations
        sentence_docs = [nlp(s) for s in sentences]
        
        # Calculate similarity matrix
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
        
        for i in range(len(sentences)):
            for j in range(i+1, len(sentences)):
                # Use spaCy's similarity
                similarity = sentence_docs[i].similarity(sentence_docs[j])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        # Cluster sentences based on similarity
        clusters = []
        used_indices = set()
        
        for i in range(len(sentences)):
            if i in used_indices:
                continue
                
            cluster = [sentences[i]]
            used_indices.add(i)
            
            # Find similar sentences
            for j in range(len(sentences)):
                if j != i and j not in used_indices and similarity_matrix[i, j] > 0.4:
                    cluster.append(sentences[j])
                    used_indices.add(j)
            
            clusters.append(cluster)
        
        # For any remaining sentences, add them to the most similar cluster
        for i in range(len(sentences)):
            if i not in used_indices:
                best_cluster = None
                best_similarity = 0
                
                for c, cluster in enumerate(clusters):
                    for sentence in cluster:
                        j = sentences.index(sentence)
                        similarity = similarity_matrix[i, j]
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_cluster = c
                
                if best_cluster is not None:
                    clusters[best_cluster].append(sentences[i])
                else:
                    clusters.append([sentences[i]])
                    
                used_indices.add(i)
        
        return clusters
    
    def _create_intro_sentence(self, sentences: List[str], query: str) -> str:
        """Create an introductory sentence for the summary."""
        # Extract query focus
        doc = nlp(query)
        focus_term = None
        
        for token in doc:
            if token.pos_ in ('NOUN', 'PROPN'):
                focus_term = token.text
                break
        
        if not focus_term:
            # Extract subject from first sentence
            if sentences:
                first_sent_doc = nlp(sentences[0])
                for token in first_sent_doc:
                    if token.dep_ == 'nsubj' and token.pos_ in ('NOUN', 'PROPN'):
                        focus_term = token.text
                        break
        
        # Create intro
        if focus_term:
            intro_templates = [
                f"This summary provides key information about {focus_term}.",
                f"The following details explain important aspects of {focus_term}.",
                f"Here's a concise overview of {focus_term}.",
            ]
            return np.random.choice(intro_templates)
        else:
            return "Here's a summary of the relevant information:"
    
    def _create_conclusion_sentence(self, sentences: List[str], query: str) -> str:
        """Create a concluding sentence for the summary."""
        # Only create conclusion if we have enough content
        if len(sentences) < 3:
            return ""
            
        # Extract query focus
        doc = nlp(query)
        focus_terms = []
        
        for token in doc:
            if token.pos_ in ('NOUN', 'PROPN') and not token.is_stop:
                focus_terms.append(token.text)
        
        if focus_terms:
            focus_term = focus_terms[0]
            conclusion_templates = [
                f"These are the key points about {focus_term} from the available documents.",
                f"The information above summarizes the most important details about {focus_term}.",
                f"This overview provides the essential information regarding {focus_term}.",
            ]
            return np.random.choice(conclusion_templates)
        else:
            return "This summary represents the most relevant information from the available documents."


class RAGSystem:
    """Retrieval-Augmented Generation system for document Q&A."""
    
    def __init__(self, knowledge_base_path: str):
        """Initialize the RAG system."""
        self.document_processor = DocumentProcessor(knowledge_base_path)
        self.summary_generator = SummaryGenerator()
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query and generate a response."""
        print(f"Processing query: {query}")
        
        # Load documents if not already loaded
        if not self.document_processor.documents:
            self.document_processor.load_all_documents()
        
        # Determine query type
        is_question = self._is_question(query)
        print(f"Query identified as {'question' if is_question else 'keyword search'}")
        
        # Retrieve relevant documents
        if is_question:
            retrieved_docs = self.document_processor.search_documents(query, top_k=3)
            documents = [(doc_name, content) for doc_name, score, content in retrieved_docs]
        else:
            # For keywords, try exact matching
            keywords = query.split()
            # Use the longest keyword for exact matches
            keywords.sort(key=len, reverse=True)
            
            documents = []
            for keyword in keywords[:2]:  # Try with top 2 longest keywords
                if len(keyword) > 3:  # Only use meaningful keywords
                    matches = self.document_processor.keyword_search(keyword)
                    for doc_name, content in matches:
                        if doc_name not in [d for d, _ in documents]:
                            documents.append((doc_name, content))
            
            # If no exact matches, fall back to semantic search
            if not documents:
                retrieved_docs = self.document_processor.search_documents(query, top_k=3)
                documents = [(doc_name, content) for doc_name, score, content in retrieved_docs]
        
        if not documents:
            return {
                "success": False,
                "error": "No relevant information found.",
                "query": query,
                "retrieved_docs": []
            }
        
        print(f"Found {len(documents)} relevant documents")
        
        # Extract texts for summarization
        texts = [content for _, content in documents]
        
        # Generate concise summary
        summary = self.summary_generator.generate_summary(texts, query)
        
        # Extract key terms for additional context
        all_content = " ".join(texts)
        key_terms = self.document_processor.extract_key_terms(all_content)
        
        # Structure the final response
        response = self._format_response(summary, key_terms, query)
        
        return {
            "success": True,
            "query": query,
            "response": response,
            "retrieved_docs": [doc_name for doc_name, _ in documents]
        }
    
    def _is_question(self, text: str) -> bool:
        """Determine if the text is a question."""
        # Check for question marks
        if '?' in text:
            return True
            
        # Check for question words
        question_starters = ['what', 'who', 'where', 'when', 'why', 'how', 'can', 'could', 'would', 'should', 'is', 'are', 'do', 'does', 'tell', 'explain', 'describe']
        first_word = text.lower().split()[0] if text else ""
        
        return first_word in question_starters
    
    def _format_response(self, summary: str, key_terms: List[str], query: str) -> str:
        """Format the final response with summary and key terms."""
        response = f"# Summary: {query}\n\n"
        response += summary
        
        if key_terms:
            response += "\n\n## Key Terms\n"
            for term in key_terms:
                # Capitalize the first letter of each term
                formatted_term = term[0].upper() + term[1:] if term else ""
                response += f"- {formatted_term}\n"
        
        return response
    
    def generate_pdf_report(self, query: str, response: str, retrieved_docs: List[str]) -> str:
        """Generate a PDF report of the response."""
        # Create a temporary directory for the PDF
        with tempfile.TemporaryDirectory() as temp_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(temp_dir, f"response_{timestamp}.pdf")
            
            # Create a PDF document
            doc = SimpleDocTemplate(
                output_path,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            # Create styles
            styles = getSampleStyleSheet()
            styles.add(ParagraphStyle(
                name='Justify',
                fontName='Helvetica',
                fontSize=10,
                leading=14,
                alignment=TA_JUSTIFY
            ))
            
            # Create elements for the PDF
            elements = []
            
            # Add title
            title_style = styles['Heading1']
            title_style.alignment = TA_LEFT
            elements.append(Paragraph(f"Summary: {query}", title_style))
            elements.append(Spacer(1, 0.25 * inch))
            
            # Process markdown in response
            paragraphs = response.split("\n\n")
            for paragraph in paragraphs:
                if paragraph.startswith("# "):
                    # Main heading
                    heading_text = paragraph[2:].strip()
                    elements.append(Paragraph(heading_text, styles['Heading1']))
                    elements.append(Spacer(1, 0.15 * inch))
                elif paragraph.startswith("## "):
                    # Subheading
                    subheading_text = paragraph[3:].strip()
                    elements.append(Paragraph(subheading_text, styles['Heading2']))
                    elements.append(Spacer(1, 0.1 * inch))
                elif paragraph.startswith("- "):
                    # List item
                    list_text = paragraph[2:].strip()
                    elements.append(Paragraph("• " + list_text, styles['Normal']))
                    elements.append(Spacer(1, 0.05 * inch))
                else:
                    # Regular paragraph
                    elements.append(Paragraph(paragraph, styles['Justify']))
                    elements.append(Spacer(1, 0.1 * inch))
            
            # Add sources
            elements.append(Paragraph("Sources", styles['Heading2']))
            elements.append(Spacer(1, 0.1 * inch))
            
            for doc in retrieved_docs:
                elements.append(Paragraph("• " + doc, styles['Normal']))
                elements.append(Spacer(1, 0.05 * inch))
            
            # Add footer with timestamp
            footer_text = f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            elements.append(Spacer(1, 0.5 * inch))
            elements.append(Paragraph(footer_text, styles['Normal']))
            
            # Build the PDF
            doc.build(elements)
            
            # In Vertex AI Workbench notebook environment
            final_path = f"/home/jupyter/response_{timestamp}.pdf"
            os.system(f"cp {output_path} {final_path}")
            
            return final_path


# Main function to run the RAG system
def main():
    """Main function to run the RAG system."""
    KNOWLEDGE_BASE_PATH = "knowledge_base_docs"
    
    print(f"Initializing RAG system with knowledge base path: {KNOWLEDGE_BASE_PATH}")
    
    # Initialize the RAG system
    rag_system = RAGSystem(KNOWLEDGE_BASE_PATH)
    
    # Example usage
    while True:
        query = input("\nEnter your query (or 'exit' to quit): ")
        
        if query.lower() == 'exit':
            break
            
        print("\nProcessing your query...\n")
        
        # Process the query
        result = rag_system.process_query(query)
        
        if result["success"]:
            print("=" * 80)
            print("RESPONSE:")
            print("-" * 80)
            print(result["response"])
            print("=" * 80)
            print("\nSources:", ", ".join(result["retrieved_docs"]))
            
            # Generate PDF report
            try:
                pdf_path = rag_system.generate_pdf_report(
                    query, 
                    result["response"], 
                    result["retrieved_docs"]
                )
                print(f"\nPDF report generated at: {pdf_path}")
            except Exception as e:
                print(f"\nError generating PDF report: {str(e)}")
        else:
            print(f"Error: {result['error']}")


# For use in a Jupyter notebook
def create_rag_system():
    """Create and return a RAG system instance for use in a notebook."""
    KNOWLEDGE_BASE_PATH = "knowledge_base_docs"
    return RAGSystem(KNOWLEDGE_BASE_PATH)


def process_query_and_generate_pdf(rag_system, query):
    """Process a query and generate a PDF report."""
    result = rag_system.process_query(query)
    
    if result["success"]:
        print("=" * 80)
        print("RESPONSE:")
        print("-" * 80)
        print(result["response"])
        print("=" * 80)
        print("\nSources:", ", ".join(result["retrieved_docs"]))
        
        # Generate PDF report
        try:
            pdf_path = rag_system.generate_pdf_report(
                query, 
                result["response"], 
                result["retrieved_docs"]
            )
            print(f"\nPDF report generated at: {pdf_path}")
            return result["response"], pdf_path
        except Exception as e:
            print(f"\nError generating PDF report: {str(e)}")
            return result["response"], None
    else:
        print(f"Error: {result['error']}")
        return None, None


# Run the system if executed directly
if __name__ == "__main__":
    main()
