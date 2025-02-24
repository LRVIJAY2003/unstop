# unstop
includes algorithms that will be scaled as per the requirements
pip install numpy pandas nltk spacy scikit-learn reportlab python-docx PyPDF2 google-cloud-aiplatform
pip install google-generativeai
python -m spacy download en_core_web_sm
gcloud auth application-default login
pip install vertexai
import os
import re
import glob
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path
import tempfile
import docx
import PyPDF2
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime
import google.generativeai as genai
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel
from vertexai.preview.generative_models import GenerativeModel as PreviewGenerativeModel

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # If model not installed, download it
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class DocumentProcessor:
    """Handles document processing, text extraction, and embedding generation."""
    
    def __init__(self, knowledge_base_path: str):
        """
        Initialize the document processor.
        
        Args:
            knowledge_base_path: Path to the folder containing knowledge base documents
        """
        self.knowledge_base_path = knowledge_base_path
        self.documents = {}  # Will store document content
        self.doc_embeddings = {}  # Will store document embeddings
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
    def load_all_documents(self) -> Dict[str, str]:
        """
        Load all documents from the knowledge base.
        
        Returns:
            Dictionary mapping document names to their content
        """
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
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better matching.
        
        Args:
            text: Text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Process with spaCy for better tokenization and lemmatization
        doc = nlp(text)
        
        # Get lemmatized tokens, excluding stopwords and punctuation
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        
        return ' '.join(tokens)
    
    def search_documents(self, query: str, top_k: int = 3) -> List[Tuple[str, float, str]]:
        """
        Search for documents relevant to the query.
        
        Args:
            query: User query
            top_k: Number of top results to return
            
        Returns:
            List of tuples (doc_name, similarity_score, content)
        """
        if not self.doc_embeddings:
            self.create_document_embeddings()
        
        # Preprocess the query
        processed_query = self.preprocess_text(query)
        
        # Transform query using the same vectorizer
        query_vector = self.vectorizer.transform([processed_query])
        
        results = []
        
        # Calculate similarity between query and all documents
        for doc_name, doc_vector in self.doc_embeddings.items():
            similarity = cosine_similarity(query_vector, doc_vector)[0][0]
            results.append((doc_name, similarity, self.documents[doc_name]))
            
        # Sort by similarity (highest first) and return top_k results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def exact_keyword_search(self, keyword: str) -> List[Tuple[str, str]]:
        """
        Search for exact keyword matches in documents.
        
        Args:
            keyword: Keyword to search for
            
        Returns:
            List of tuples (doc_name, relevant_context)
        """
        if not self.documents:
            self.load_all_documents()
            
        results = []
        keyword_lower = keyword.lower()
        
        for doc_name, content in self.documents.items():
            # Check if keyword exists in the document
            if keyword_lower in content.lower():
                # Extract context around the keyword
                relevant_context = self.extract_context(content, keyword_lower)
                if relevant_context:
                    results.append((doc_name, relevant_context))
                    
        return results
    
    def extract_context(self, text: str, keyword: str, context_size: int = 200) -> str:
        """
        Extract context around a keyword in text.
        
        Args:
            text: Full text to extract from
            keyword: Keyword to find
            context_size: Number of characters to extract around the keyword
            
        Returns:
            Text with context around keyword
        """
        text_lower = text.lower()
        keyword_lower = keyword.lower()
        
        # Find all occurrences of the keyword
        matches = [match.start() for match in re.finditer(keyword_lower, text_lower)]
        
        if not matches:
            return ""
            
        # Extract context around first occurrence
        start_pos = max(0, matches[0] - context_size)
        end_pos = min(len(text), matches[0] + len(keyword) + context_size)
        
        return text[start_pos:end_pos]


class RAGSystem:
    """Implements the Retrieval-Augmented Generation system."""
    
    def __init__(self, knowledge_base_path: str, project_id: str, location: str = "us-central1"):
        """
        Initialize the RAG system.
        
        Args:
            knowledge_base_path: Path to the folder containing knowledge base documents
            project_id: Google Cloud project ID
            location: Google Cloud region
        """
        self.project_id = project_id
        self.location = location
        self.document_processor = DocumentProcessor(knowledge_base_path)
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=location)
        
        # Set up Gemini model
        try:
            self.model = GenerativeModel(model_name="gemini-1.5-pro")
        except Exception as e:
            print(f"Error initializing Gemini model: {str(e)}")
            # Fallback to Preview model if needed
            try:
                self.model = PreviewGenerativeModel(model_name="gemini-1.5-pro")
                print("Using Preview Generative Model instead.")
            except Exception as e2:
                print(f"Error initializing Preview Gemini model: {str(e2)}")
                raise
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query and generate a response.
        
        Args:
            query: User query text
            
        Returns:
            Dictionary with response information
        """
        # Load documents if not already loaded
        if not self.document_processor.documents:
            self.document_processor.load_all_documents()
        
        # Check if query is a simple keyword or a complete question
        is_question = self._is_question(query)
        
        retrieved_docs = []
        
        if is_question:
            # For questions, use semantic search to find relevant documents
            results = self.document_processor.search_documents(query)
            retrieved_docs = [(doc_name, content) for doc_name, score, content in results if score > 0.1]
        else:
            # For keywords, use exact matching first
            retrieved_docs = self.document_processor.exact_keyword_search(query)
            
            # If no exact matches, fall back to semantic search
            if not retrieved_docs:
                results = self.document_processor.search_documents(query)
                retrieved_docs = [(doc_name, content) for doc_name, score, content in results if score > 0.1]
        
        if not retrieved_docs:
            # No relevant documents found
            return {
                "success": False,
                "error": "No relevant information found for your query.",
                "query": query,
                "retrieved_docs": []
            }
        
        # Generate response using Gemini
        response = self._generate_summary(query, retrieved_docs)
        
        return {
            "success": True,
            "query": query,
            "response": response,
            "retrieved_docs": [doc_name for doc_name, _ in retrieved_docs]
        }
    
    def _is_question(self, text: str) -> bool:
        """
        Determine if text is a question.
        
        Args:
            text: Text to analyze
            
        Returns:
            True if text appears to be a question
        """
        # Check for question marks
        if '?' in text:
            return True
            
        # Check for question words
        question_starters = ['what', 'who', 'where', 'when', 'why', 'how', 'can', 'could', 'would', 'should', 'is', 'are', 'do', 'does']
        first_word = text.lower().split()[0] if text else ""
        
        return first_word in question_starters
    
    def _generate_summary(self, query: str, retrieved_docs: List[Tuple[str, str]]) -> str:
        """
        Generate a summary response using Gemini.
        
        Args:
            query: User query
            retrieved_docs: List of (doc_name, content) tuples
            
        Returns:
            Generated summary
        """
        # Prepare prompt for Gemini
        context = "\n\n".join([f"Document: {doc_name}\nContent: {content}" for doc_name, content in retrieved_docs])
        
        prompt = f"""
        You are an AI assistant tasked with providing accurate information based on the documents I provide.
        
        User Query: {query}
        
        Context from relevant documents:
        {context}
        
        Please provide a clear and concise answer to the query based ONLY on the information in these documents.
        Do not hallucinate or add information not present in the documents.
        If the documents don't contain enough information to answer the query fully, acknowledge that limitation.
        
        Your response should be well-structured and directly address the user's query.
        """
        
        try:
            # Generate response
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating content with Gemini: {str(e)}")
            # Fallback solution - extract key sentences as summary
            all_content = " ".join([content for _, content in retrieved_docs])
            sentences = nltk.sent_tokenize(all_content)
            
            # Return first 5 sentences as a basic summary
            basic_summary = " ".join(sentences[:5])
            return f"(Note: AI summary generation failed, showing extracted content)\n\n{basic_summary}"
    
    def generate_pdf_report(self, query: str, response: str, retrieved_docs: List[str]) -> str:
        """
        Generate a PDF report of the response.
        
        Args:
            query: User query
            response: Generated response
            retrieved_docs: List of document names used
            
        Returns:
            Path to the generated PDF
        """
        # Create a temporary directory for the PDF
        with tempfile.TemporaryDirectory() as temp_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(temp_dir, f"response_{timestamp}.pdf")
            
            # Create the PDF
            c = canvas.Canvas(output_path, pagesize=letter)
            width, height = letter
            
            # Add title
            c.setFont("Helvetica-Bold", 16)
            c.drawString(72, height - 72, "Query Response Summary")
            
            # Add query
            c.setFont("Helvetica-Bold", 12)
            c.drawString(72, height - 100, "User Query:")
            c.setFont("Helvetica", 12)
            
            # Handle long queries by wrapping text
            text_object = c.beginText(72, height - 120)
            text_object.setFont("Helvetica", 12)
            
            wrapped_query = self._wrap_text(query, 80)
            for line in wrapped_query:
                text_object.textLine(line)
            
            c.drawText(text_object)
            
            # Add response with wrapped text
            y_position = height - 120 - (len(wrapped_query) * 15) - 30
            c.setFont("Helvetica-Bold", 12)
            c.drawString(72, y_position, "Response:")
            
            text_object = c.beginText(72, y_position - 20)
            text_object.setFont("Helvetica", 12)
            
            wrapped_response = self._wrap_text(response, 80)
            for line in wrapped_response:
                text_object.textLine(line)
            
            c.drawText(text_object)
            
            # Add sources
            y_position = y_position - 20 - (len(wrapped_response) * 15) - 30
            
            if y_position < 72:  # If we're running out of space, add a new page
                c.showPage()
                y_position = height - 72
            
            c.setFont("Helvetica-Bold", 12)
            c.drawString(72, y_position, "Sources:")
            
            text_object = c.beginText(72, y_position - 20)
            text_object.setFont("Helvetica", 10)
            
            for doc in retrieved_docs:
                text_object.textLine(f"- {doc}")
            
            c.drawText(text_object)
            
            # Add timestamp
            c.setFont("Helvetica", 10)
            c.drawString(72, 72, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            c.save()
            
            # In a real environment, you would save this to a persistent location
            # For this example, we'll return the path, but in Vertex AI Workbench
            # you might want to use Google Cloud Storage
            
            return output_path
    
    def _wrap_text(self, text: str, width: int) -> List[str]:
        """
        Wrap text to fit within specified width.
        
        Args:
            text: Text to wrap
            width: Maximum line width in characters
            
        Returns:
            List of wrapped lines
        """
        lines = []
        for paragraph in text.split('\n'):
            if len(paragraph) <= width:
                lines.append(paragraph)
            else:
                words = paragraph.split()
                current_line = []
                current_length = 0
                
                for word in words:
                    if current_length + len(word) + len(current_line) <= width:
                        current_line.append(word)
                        current_length += len(word)
                    else:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                        current_length = len(word)
                
                if current_line:
                    lines.append(' '.join(current_line))
        
        return lines


def main():
    """Main function to run the RAG system."""
    
    # Configuration parameters
    KNOWLEDGE_BASE_PATH = "knowledge_base_docs"  # Update with your actual path
    PROJECT_ID = "your-project-id"  # Update with your GCP project ID
    
    # Initialize the RAG system
    rag_system = RAGSystem(KNOWLEDGE_BASE_PATH, PROJECT_ID)
    
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


if __name__ == "__main__":
    main()

import os
import re
import glob
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path
import tempfile
import docx
import PyPDF2
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime
import google.generativeai as genai
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel, Part

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # If model not installed, download it
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class DocumentProcessor:
    """Handles document processing, text extraction, and embedding generation."""
    
    def __init__(self, knowledge_base_path: str):
        """
        Initialize the document processor.
        
        Args:
            knowledge_base_path: Path to the folder containing knowledge base documents
        """
        self.knowledge_base_path = knowledge_base_path
        self.documents = {}  # Will store document content
        self.doc_embeddings = {}  # Will store document embeddings
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
    def load_all_documents(self) -> Dict[str, str]:
        """
        Load all documents from the knowledge base.
        
        Returns:
            Dictionary mapping document names to their content
        """
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
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better matching.
        
        Args:
            text: Text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Process with spaCy for better tokenization and lemmatization
        doc = nlp(text)
        
        # Get lemmatized tokens, excluding stopwords and punctuation
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        
        return ' '.join(tokens)
    
    def search_documents(self, query: str, top_k: int = 3) -> List[Tuple[str, float, str]]:
        """
        Search for documents relevant to the query.
        
        Args:
            query: User query
            top_k: Number of top results to return
            
        Returns:
            List of tuples (doc_name, similarity_score, content)
        """
        if not self.doc_embeddings:
            self.create_document_embeddings()
        
        # Preprocess the query
        processed_query = self.preprocess_text(query)
        
        # Transform query using the same vectorizer
        query_vector = self.vectorizer.transform([processed_query])
        
        results = []
        
        # Calculate similarity between query and all documents
        for doc_name, doc_vector in self.doc_embeddings.items():
            similarity = cosine_similarity(query_vector, doc_vector)[0][0]
            results.append((doc_name, similarity, self.documents[doc_name]))
            
        # Sort by similarity (highest first) and return top_k results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def exact_keyword_search(self, keyword: str) -> List[Tuple[str, str]]:
        """
        Search for exact keyword matches in documents.
        
        Args:
            keyword: Keyword to search for
            
        Returns:
            List of tuples (doc_name, relevant_context)
        """
        if not self.documents:
            self.load_all_documents()
            
        results = []
        keyword_lower = keyword.lower()
        
        for doc_name, content in self.documents.items():
            # Check if keyword exists in the document
            if keyword_lower in content.lower():
                # Extract context around the keyword
                relevant_context = self.extract_context(content, keyword_lower)
                if relevant_context:
                    results.append((doc_name, relevant_context))
                    
        return results
    
    def extract_context(self, text: str, keyword: str, context_size: int = 500) -> str:
        """
        Extract context around a keyword in text.
        
        Args:
            text: Full text to extract from
            keyword: Keyword to find
            context_size: Number of characters to extract around the keyword
            
        Returns:
            Text with context around keyword
        """
        text_lower = text.lower()
        keyword_lower = keyword.lower()
        
        # Find all occurrences of the keyword
        matches = [match.start() for match in re.finditer(keyword_lower, text_lower)]
        
        if not matches:
            return ""
            
        # Extract context around first occurrence
        start_pos = max(0, matches[0] - context_size)
        end_pos = min(len(text), matches[0] + len(keyword) + context_size)
        
        # Find sentence boundaries if possible
        if start_pos > 0:
            # Try to start at the beginning of a sentence
            sentence_start = text.rfind('.', 0, start_pos)
            if sentence_start != -1:
                start_pos = sentence_start + 1
        
        if end_pos < len(text):
            # Try to end at the end of a sentence
            sentence_end = text.find('.', end_pos)
            if sentence_end != -1:
                end_pos = sentence_end + 1
        
        return text[start_pos:end_pos].strip()


class RAGSystem:
    """Implements the Retrieval-Augmented Generation system."""
    
    def __init__(self, knowledge_base_path: str, project_id: str, location: str = "us-central1"):
        """
        Initialize the RAG system.
        
        Args:
            knowledge_base_path: Path to the folder containing knowledge base documents
            project_id: Google Cloud project ID
            location: Google Cloud region
        """
        self.project_id = project_id
        self.location = location
        self.document_processor = DocumentProcessor(knowledge_base_path)
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=location)
        
        # Set up Gemini model
        try:
            print("Initializing Gemini model...")
            self.model = GenerativeModel("gemini-1.5-pro")
            print("Gemini model initialized successfully")
        except Exception as e:
            print(f"Error initializing Gemini model: {str(e)}")
            raise
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query and generate a response.
        
        Args:
            query: User query text
            
        Returns:
            Dictionary with response information
        """
        print(f"Processing query: {query}")
        
        # Load documents if not already loaded
        if not self.document_processor.documents:
            self.document_processor.load_all_documents()
        
        # Check if query is a simple keyword or a complete question
        is_question = self._is_question(query)
        print(f"Query identified as {'question' if is_question else 'keyword search'}")
        
        retrieved_docs = []
        
        if is_question:
            # For questions, use semantic search to find relevant documents
            results = self.document_processor.search_documents(query)
            retrieved_docs = [(doc_name, content) for doc_name, score, content in results if score > 0.1]
        else:
            # For keywords, use exact matching first
            retrieved_docs = self.document_processor.exact_keyword_search(query)
            
            # If no exact matches, fall back to semantic search
            if not retrieved_docs:
                print("No exact matches found, falling back to semantic search")
                results = self.document_processor.search_documents(query)
                retrieved_docs = [(doc_name, content) for doc_name, score, content in results if score > 0.1]
        
        print(f"Found {len(retrieved_docs)} relevant documents")
        
        if not retrieved_docs:
            # No relevant documents found
            return {
                "success": False,
                "error": "No relevant information found for your query.",
                "query": query,
                "retrieved_docs": []
            }
        
        # Generate response using Gemini
        response = self._generate_summary(query, retrieved_docs)
        
        return {
            "success": True,
            "query": query,
            "response": response,
            "retrieved_docs": [doc_name for doc_name, _ in retrieved_docs]
        }
    
    def _is_question(self, text: str) -> bool:
        """
        Determine if text is a question.
        
        Args:
            text: Text to analyze
            
        Returns:
            True if text appears to be a question
        """
        # Check for question marks
        if '?' in text:
            return True
            
        # Check for question words
        question_starters = ['what', 'who', 'where', 'when', 'why', 'how', 'can', 'could', 'would', 'should', 'is', 'are', 'do', 'does', 'tell', 'explain', 'describe']
        first_word = text.lower().split()[0] if text else ""
        
        return first_word in question_starters
    
    def _generate_summary(self, query: str, retrieved_docs: List[Tuple[str, str]]) -> str:
        """
        Generate a summary response using Gemini.
        
        Args:
            query: User query
            retrieved_docs: List of (doc_name, content) tuples
            
        Returns:
            Generated summary
        """
        # Prepare context from documents
        context_parts = []
        for i, (doc_name, content) in enumerate(retrieved_docs):
            # Limit context length to avoid token limits
            if len(content) > 8000:
                content = content[:8000] + "..."
            
            context_parts.append(f"Document {i+1} ({doc_name}):\n{content}\n")
        
        context = "\n".join(context_parts)
        
        # Prepare prompt for Gemini
        prompt = f"""
        You are an AI assistant tasked with providing accurate information based on the documents I provide.
        
        User Query: {query}
        
        Context from relevant documents:
        {context}
        
        Please provide a clear and concise answer to the query based ONLY on the information in these documents.
        Do not hallucinate or add information not present in the documents.
        If the documents don't contain enough information to answer the query fully, acknowledge that limitation.
        
        Your response should be well-structured and directly address the user's query.
        Include specific details from the documents to support your answer.
        """
        
        try:
            print("Generating response with Gemini...")
            # Generate response
            response = self.model.generate_content(prompt)
            print("Response generated successfully")
            return response.text
        except Exception as e:
            print(f"Error generating content with Gemini: {str(e)}")
            try:
                # Try with a shorter context if it might be a token limit issue
                shortened_context = "\n".join([f"Document {i+1} ({doc_name}):\n{content[:1000]}..." 
                                              for i, (doc_name, content) in enumerate(retrieved_docs)])
                
                shorter_prompt = f"""
                You are an AI assistant tasked with providing accurate information based on the documents I provide.
                
                User Query: {query}
                
                Context from relevant documents (shortened):
                {shortened_context}
                
                Please provide a concise answer based ONLY on the information in these documents.
                """
                
                print("Trying with shorter context...")
                response = self.model.generate_content(shorter_prompt)
                print("Response generated successfully with shorter context")
                return response.text
            except Exception as e2:
                print(f"Second error generating content: {str(e2)}")
                # Fallback solution - extract key sentences as summary
                all_content = " ".join([content for _, content in retrieved_docs])
                sentences = nltk.sent_tokenize(all_content)
                
                # Return first 5 sentences as a basic summary
                basic_summary = " ".join(sentences[:5])
                return f"(Note: AI summary generation failed, showing extracted content)\n\n{basic_summary}"
    
    def generate_pdf_report(self, query: str, response: str, retrieved_docs: List[str]) -> str:
        """
        Generate a PDF report of the response.
        
        Args:
            query: User query
            response: Generated response
            retrieved_docs: List of document names used
            
        Returns:
            Path to the generated PDF
        """
        # Create a temporary directory for the PDF
        with tempfile.TemporaryDirectory() as temp_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(temp_dir, f"response_{timestamp}.pdf")
            
            # Create the PDF
            c = canvas.Canvas(output_path, pagesize=letter)
            width, height = letter
            
            # Add title
            c.setFont("Helvetica-Bold", 16)
            c.drawString(72, height - 72, "Query Response Summary")
            
            # Add query
            c.setFont("Helvetica-Bold", 12)
            c.drawString(72, height - 100, "User Query:")
            c.setFont("Helvetica", 12)
            
            # Handle long queries by wrapping text
            text_object = c.beginText(72, height - 120)
            text_object.setFont("Helvetica", 12)
            
            wrapped_query = self._wrap_text(query, 80)
            for line in wrapped_query:
                text_object.textLine(line)
            
            c.drawText(text_object)
            
            # Add response with wrapped text
            y_position = height - 120 - (len(wrapped_query) * 15) - 30
            c.setFont("Helvetica-Bold", 12)
            c.drawString(72, y_position, "Response:")
            
            text_object = c.beginText(72, y_position - 20)
            text_object.setFont("Helvetica", 12)
            
            wrapped_response = self._wrap_text(response, 80)
            for line in wrapped_response:
                text_object.textLine(line)
            
            c.drawText(text_object)
            
            # Add sources
            y_position = y_position - 20 - (len(wrapped_response) * 15) - 30
            
            if y_position < 72:  # If we're running out of space, add a new page
                c.showPage()
                y_position = height - 72
            
            c.setFont("Helvetica-Bold", 12)
            c.drawString(72, y_position, "Sources:")
            
            text_object = c.beginText(72, y_position - 20)
            text_object.setFont("Helvetica", 10)
            
            for doc in retrieved_docs:
                text_object.textLine(f"- {doc}")
            
            c.drawText(text_object)
            
            # Add timestamp
            c.setFont("Helvetica", 10)
            c.drawString(72, 72, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            c.save()
            
            # In Vertex AI Workbench notebook environment
            final_path = f"/home/jupyter/response_{timestamp}.pdf"
            os.system(f"cp {output_path} {final_path}")
            
            return final_path
    
    def _wrap_text(self, text: str, width: int) -> List[str]:
        """
        Wrap text to fit within specified width.
        
        Args:
            text: Text to wrap
            width: Maximum line width in characters
            
        Returns:
            List of wrapped lines
        """
        lines = []
        for paragraph in text.split('\n'):
            if len(paragraph) <= width:
                lines.append(paragraph)
            else:
                words = paragraph.split()
                current_line = []
                current_length = 0
                
                for word in words:
                    if current_length + len(word) + len(current_line) <= width:
                        current_line.append(word)
                        current_length += len(word)
                    else:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                        current_length = len(word)
                
                if current_line:
                    lines.append(' '.join(current_line))
        
        return lines


# Main function to run the RAG system directly
def main():
    """Main function to run the RAG system."""
    
    # Configuration parameters - using your specified values
    KNOWLEDGE_BASE_PATH = "knowledge_base_docs"
    PROJECT_ID = "cloud-workspace-poc-51731"
    LOCATION = "us-central1"
    
    print(f"Initializing RAG system with:")
    print(f"- Knowledge base path: {KNOWLEDGE_BASE_PATH}")
    print(f"- Project ID: {PROJECT_ID}")
    print(f"- Location: {LOCATION}")
    
    # Initialize the RAG system
    rag_system = RAGSystem(KNOWLEDGE_BASE_PATH, PROJECT_ID, LOCATION)
    
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
    PROJECT_ID = "cloud-workspace-poc-51731"
    LOCATION = "us-central1"
    
    return RAGSystem(KNOWLEDGE_BASE_PATH, PROJECT_ID, LOCATION)


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


if __name__ == "__main__":
    main()
