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
import heapq
from collections import Counter

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
    """Implements the Retrieval-Augmented Generation system without external APIs."""
    
    def __init__(self, knowledge_base_path: str):
        """
        Initialize the RAG system.
        
        Args:
            knowledge_base_path: Path to the folder containing knowledge base documents
        """
        self.document_processor = DocumentProcessor(knowledge_base_path)
    
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
        
        # Generate response using extractive summarization
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
        Generate a summary response using extractive summarization.
        
        Args:
            query: User query
            retrieved_docs: List of (doc_name, content) tuples
            
        Returns:
            Generated summary
        """
        # First, try to extract sentences that are most relevant to the query
        query_sentences = self._extract_query_relevant_sentences(query, retrieved_docs)
        
        # Also extract key sentences from the documents
        key_sentences = self._extract_key_sentences(retrieved_docs)
        
        # Combine the sentences, prioritizing query-relevant ones
        all_summary_sentences = query_sentences + [s for s in key_sentences if s not in query_sentences]
        
        # Limit to a reasonable number of sentences
        summary_sentences = all_summary_sentences[:10]
        
        # Format the response
        if summary_sentences:
            summary = " ".join(summary_sentences)
            
            # Add an introduction
            intro = f"Based on the information in the documents, here's what I found about '{query}':\n\n"
            
            # Add document sources
            sources = "\n\nInformation sourced from: " + ", ".join([doc_name for doc_name, _ in retrieved_docs])
            
            return intro + summary + sources
        else:
            # Fallback if no good sentences were found
            return self._generate_simple_summary(retrieved_docs)
    
    def _extract_query_relevant_sentences(self, query: str, retrieved_docs: List[Tuple[str, str]]) -> List[str]:
        """
        Extract sentences that are most relevant to the query.
        
        Args:
            query: User query
            retrieved_docs: List of (doc_name, content) tuples
            
        Returns:
            List of relevant sentences
        """
        # Preprocess the query
        processed_query = self.document_processor.preprocess_text(query)
        query_terms = processed_query.split()
        
        if not query_terms:
            return []
        
        # Extract sentences from all documents
        all_sentences = []
        for _, content in retrieved_docs:
            sentences = nltk.sent_tokenize(content)
            all_sentences.extend(sentences)
        
        # Score sentences based on query term matches
        sentence_scores = []
        for sentence in all_sentences:
            # Clean and tokenize sentence
            clean_sentence = self.document_processor.preprocess_text(sentence)
            sentence_terms = clean_sentence.split()
            
            # Count matching terms
            matches = sum(1 for term in query_terms if term in sentence_terms)
            if matches > 0:
                # Score is the percentage of query terms that match
                score = matches / len(query_terms)
                sentence_scores.append((sentence, score))
        
        # Sort by score (highest first)
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take top sentences
        return [sentence for sentence, score in sentence_scores[:5] if score > 0.2]
    
    def _extract_key_sentences(self, retrieved_docs: List[Tuple[str, str]]) -> List[str]:
        """
        Extract key sentences from documents using TextRank-like algorithm.
        
        Args:
            retrieved_docs: List of (doc_name, content) tuples
            
        Returns:
            List of key sentences
        """
        # Extract all sentences
        all_sentences = []
        for _, content in retrieved_docs:
            sentences = nltk.sent_tokenize(content)
            all_sentences.extend([s for s in sentences if len(s.split()) > 5])  # Filter very short sentences
        
        if not all_sentences:
            return []
        
        # Create sentence vectors
        sentence_vectors = {}
        for sentence in all_sentences:
            clean_sentence = self.document_processor.preprocess_text(sentence)
            words = clean_sentence.split()
            sentence_vectors[sentence] = Counter(words)
        
        # Calculate similarity between sentences
        sentence_scores = {}
        for sentence in all_sentences:
            sentence_scores[sentence] = 0
            for other_sentence in all_sentences:
                if sentence != other_sentence:
                    # Simple intersection of word sets as similarity measure
                    common_words = set(sentence_vectors[sentence].keys()) & set(sentence_vectors[other_sentence].keys())
                    similarity = len(common_words) / (len(sentence_vectors[sentence]) + len(sentence_vectors[other_sentence]) + 1e-10)
                    sentence_scores[sentence] += similarity
        
        # Get top sentences
        top_sentences = heapq.nlargest(5, sentence_scores.items(), key=lambda x: x[1])
        
        # Sort sentences by their original order
        ordered_sentences = [s[0] for s in top_sentences]
        ordered_sentences.sort(key=lambda s: all_sentences.index(s))
        
        return ordered_sentences
    
    def _generate_simple_summary(self, retrieved_docs: List[Tuple[str, str]]) -> str:
        """
        Generate a simple summary from the documents.
        
        Args:
            retrieved_docs: List of (doc_name, content) tuples
            
        Returns:
            Simple summary text
        """
        summary = "Here's information found in the documents:\n\n"
        
        for doc_name, content in retrieved_docs:
            # Extract first few sentences from each document
            sentences = nltk.sent_tokenize(content)
            doc_summary = " ".join(sentences[:3])  # First 3 sentences
            
            summary += f"From {doc_name}:\n{doc_summary}\n\n"
            
        return summary
    
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
    
    print(f"Initializing RAG system with:")
    print(f"- Knowledge base path: {KNOWLEDGE_BASE_PATH}")
    
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


if __name__ == "__main__":
    main()

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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib import colors
from datetime import datetime
import heapq
from collections import Counter, defaultdict
import string
import itertools
import textwrap

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

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
        self.document_sections = {}  # Will store sections of documents
        self.doc_embeddings = {}  # Will store document embeddings
        self.section_embeddings = {}  # Will store section embeddings
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams for better matching
            max_df=0.85,         # Ignore terms that appear in >85% of docs
            min_df=2,            # Ignore terms that appear in <2 docs
        )
        self.count_vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2))
        
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
        
        # Create document sections for more granular retrieval
        self._create_document_sections()
        
        return self.documents
    
    def _create_document_sections(self):
        """Create sections from the documents for more granular retrieval."""
        for doc_name, content in self.documents.items():
            # Split documents into paragraphs
            paragraphs = [p for p in content.split('\n\n') if p.strip()]
            
            # Group paragraphs into coherent sections (at most 5 paragraphs per section)
            sections = []
            current_section = []
            section_length = 0
            
            for paragraph in paragraphs:
                # If paragraph is very short, combine with the previous one
                if len(paragraph.split()) < 10 and current_section:
                    current_section[-1] += " " + paragraph
                else:
                    current_section.append(paragraph)
                    section_length += 1
                
                # Start a new section if current one is getting too long
                if section_length >= 3:
                    sections.append('\n'.join(current_section))
                    current_section = []
                    section_length = 0
            
            # Add the last section if it exists
            if current_section:
                sections.append('\n'.join(current_section))
                
            # Store the sections for this document
            self.document_sections[doc_name] = sections
            
        print(f"Created {sum(len(sections) for sections in self.document_sections.values())} sections from {len(self.documents)} documents")
    
    def create_document_embeddings(self):
        """Create TF-IDF embeddings for all documents and their sections."""
        if not self.documents:
            self.load_all_documents()
            
        # Prepare document content for vectorization
        docs_content = list(self.documents.values())
        doc_names = list(self.documents.keys())
        
        # Fit and transform to get document embeddings
        tfidf_matrix = self.vectorizer.fit_transform(docs_content)
        
        # Store document embeddings with their names
        for i, doc_name in enumerate(doc_names):
            self.doc_embeddings[doc_name] = tfidf_matrix[i]
        
        # Create section embeddings
        section_texts = []
        section_ids = []
        
        for doc_name, sections in self.document_sections.items():
            for i, section in enumerate(sections):
                section_texts.append(section)
                section_ids.append((doc_name, i))
        
        if section_texts:
            # Transform the section texts using the already-fitted vectorizer
            section_matrix = self.vectorizer.transform(section_texts)
            
            # Store section embeddings
            for idx, (doc_name, section_idx) in enumerate(section_ids):
                self.section_embeddings[(doc_name, section_idx)] = section_matrix[idx]
    
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
    
    def search_sections(self, query: str, top_k: int = 5) -> List[Tuple[str, int, float, str]]:
        """
        Search for document sections relevant to the query.
        
        Args:
            query: User query
            top_k: Number of top results to return
            
        Returns:
            List of tuples (doc_name, section_idx, similarity_score, content)
        """
        if not self.section_embeddings:
            self.create_document_embeddings()
        
        # Preprocess the query
        processed_query = self.preprocess_text(query)
        
        # Transform query using the same vectorizer
        query_vector = self.vectorizer.transform([processed_query])
        
        results = []
        
        # Calculate similarity between query and all sections
        for (doc_name, section_idx), section_vector in self.section_embeddings.items():
            similarity = cosine_similarity(query_vector, section_vector)[0][0]
            content = self.document_sections[doc_name][section_idx]
            results.append((doc_name, section_idx, similarity, content))
            
        # Sort by similarity (highest first) and return top_k results
        results.sort(key=lambda x: x[2], reverse=True)
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
    
    def extract_context(self, text: str, keyword: str, context_size: int = 1000) -> str:
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
        
        # Extract contexts for all occurrences
        all_contexts = []
        for match_pos in matches[:3]:  # Limit to first 3 occurrences
            # Extract context around occurrence
            start_pos = max(0, match_pos - context_size // 2)
            end_pos = min(len(text), match_pos + len(keyword) + context_size // 2)
            
            # Find sentence boundaries if possible
            if start_pos > 0:
                # Try to start at the beginning of a sentence
                sentence_start = text.rfind('.', 0, start_pos)
                if sentence_start != -1:
                    start_pos = sentence_start + 1
            
            if end_pos < len(text):
                # Try to end at the end of a sentence
                sentence_end = text.find('.', end_pos)
                if sentence_end != -1 and sentence_end - end_pos < 200:  # Don't extend too far
                    end_pos = sentence_end + 1
            
            context = text[start_pos:end_pos].strip()
            all_contexts.append(context)
        
        # Combine all contexts
        return "\n\n".join(all_contexts)
    
    def extract_key_phrases(self, text: str, top_n: int = 5) -> List[str]:
        """
        Extract key phrases from text.
        
        Args:
            text: Text to analyze
            top_n: Number of key phrases to extract
            
        Returns:
            List of key phrases
        """
        # Process with spaCy
        doc = nlp(text)
        
        # Extract noun phrases
        noun_phrases = []
        for chunk in doc.noun_chunks:
            clean_chunk = ' '.join(token.lemma_ for token in chunk if not token.is_stop and not token.is_punct)
            if clean_chunk and len(clean_chunk.split()) <= 4:  # Limit to reasonable length phrases
                noun_phrases.append(clean_chunk)
        
        # Count phrase frequencies
        phrase_counts = Counter(noun_phrases)
        
        # Return most common phrases
        return [phrase for phrase, _ in phrase_counts.most_common(top_n)]


class RAGSystem:
    """Implements the Retrieval-Augmented Generation system without external APIs."""
    
    def __init__(self, knowledge_base_path: str):
        """
        Initialize the RAG system.
        
        Args:
            knowledge_base_path: Path to the folder containing knowledge base documents
        """
        self.document_processor = DocumentProcessor(knowledge_base_path)
    
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
        
        # Different retrieval strategies based on query type
        if is_question:
            # For questions, search both full documents and sections
            doc_results = self.document_processor.search_documents(query, top_k=2)
            section_results = self.document_processor.search_sections(query, top_k=5)
            
            # Combine document and section results
            retrieved_docs = [(doc_name, content) for doc_name, _, content in doc_results]
            for doc_name, _, _, content in section_results:
                # Avoid duplicating entire documents
                if not any(doc_name == d_name for d_name, _ in retrieved_docs):
                    retrieved_docs.append((doc_name, content))
        else:
            # For keywords, use exact matching first
            retrieved_docs = self.document_processor.exact_keyword_search(query)
            
            # If no exact matches, fall back to semantic search with sections
            if not retrieved_docs:
                print("No exact matches found, falling back to semantic search")
                section_results = self.document_processor.search_sections(query, top_k=5)
                retrieved_docs = [(doc_name, content) for doc_name, _, _, content in section_results]
        
        print(f"Found {len(retrieved_docs)} relevant document segments")
        
        if not retrieved_docs:
            # No relevant documents found
            return {
                "success": False,
                "error": "No relevant information found for your query.",
                "query": query,
                "retrieved_docs": []
            }
        
        # Generate response using extractive summarization
        response = self._generate_enhanced_summary(query, retrieved_docs)
        
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
    
    def _generate_enhanced_summary(self, query: str, retrieved_docs: List[Tuple[str, str]]) -> str:
        """
        Generate an enhanced summary response with improved formatting.
        
        Args:
            query: User query
            retrieved_docs: List of (doc_name, content) tuples
            
        Returns:
            Generated summary
        """
        # Extract key phrases from the query
        query_phrases = set(query.lower().split())
        for word in list(query_phrases):
            if word in STOPWORDS or len(word) < 3:
                query_phrases.discard(word)
        
        # Process all retrieved documents
        all_sentences = []
        doc_sentences = {}
        doc_key_phrases = {}
        
        for doc_name, content in retrieved_docs:
            # Extract sentences
            sentences = nltk.sent_tokenize(content)
            doc_sentences[doc_name] = sentences
            all_sentences.extend(sentences)
            
            # Extract key phrases
            doc_key_phrases[doc_name] = self.document_processor.extract_key_phrases(content)
        
        # Score sentences based on multiple criteria
        sentence_info = []
        
        for sentence in all_sentences:
            # Skip very short sentences
            if len(sentence.split()) < 4:
                continue
                
            # Initial score is 0
            score = 0
            
            # 1. Query term match score
            sentence_lower = sentence.lower()
            matches = sum(1 for phrase in query_phrases if phrase in sentence_lower)
            query_score = matches / max(1, len(query_phrases))
            score += query_score * 3  # Weight query matches heavily
            
            # 2. Key phrase match score
            key_phrase_matches = 0
            for doc_name, phrases in doc_key_phrases.items():
                for phrase in phrases:
                    if phrase in sentence_lower:
                        key_phrase_matches += 1
            phrase_score = min(1.0, key_phrase_matches / 5)
            score += phrase_score
            
            # 3. Sentence position score (favor sentences at the beginning of docs)
            for doc_name, sentences in doc_sentences.items():
                if sentence in sentences:
                    position = sentences.index(sentence)
                    position_score = 1.0 / (position + 1) if position < 3 else 0
                    score += position_score
            
            # 4. Sentence length score (favor medium-length sentences)
            words = len(sentence.split())
            length_score = 0
            if 8 <= words <= 25:
                length_score = 0.5
            score += length_score
            
            # Store sentence with its score and metadata
            sentence_info.append({
                'sentence': sentence,
                'score': score,
                'query_match': query_score > 0.2,
                'length': len(sentence)
            })
        
        # Sort sentences by score
        sentence_info.sort(key=lambda x: x['score'], reverse=True)
        
        # Take the top sentences (at most 15)
        top_sentences = sentence_info[:15]
        
        # Re-order sentences to maintain coherence
        # Group similar sentences using sentence clustering
        sentence_clusters = self._cluster_sentences(top_sentences)
        
        # Flatten clusters to get final ordered sentences
        final_sentences = []
        for cluster in sentence_clusters:
            final_sentences.extend([info['sentence'] for info in cluster])
        
        # Limit to a reasonable number
        if len(final_sentences) > 10:
            final_sentences = final_sentences[:10]
        
        # Format the response with proper paragraphs and sections
        if final_sentences:
            # Start with an introduction
            summary = f"# Information about: {query}\n\n"
            
            # Format the main content
            current_paragraph = []
            for i, sentence in enumerate(final_sentences):
                current_paragraph.append(sentence)
                
                # Start a new paragraph after 2-3 sentences or if the paragraph is getting long
                if (i % 3 == 2) or sum(len(s) for s in current_paragraph) > 400:
                    summary += " ".join(current_paragraph) + "\n\n"
                    current_paragraph = []
            
            # Add any remaining sentences
            if current_paragraph:
                summary += " ".join(current_paragraph) + "\n\n"
            
            # Add a conclusion with sources
            summary += "## Sources:\n"
            for doc_name, _ in retrieved_docs:
                summary += f"- {doc_name}\n"
            
            return summary
        else:
            # Fallback if no good sentences were found
            return self._generate_simple_summary(query, retrieved_docs)
    
    def _cluster_sentences(self, sentence_info: List[Dict]) -> List[List[Dict]]:
        """
        Cluster sentences by similarity to keep related sentences together.
        
        Args:
            sentence_info: List of dictionaries with sentence metadata
            
        Returns:
            List of clusters (each cluster is a list of sentence info dicts)
        """
        if len(sentence_info) <= 1:
            return [sentence_info]
        
        # Extract sentences
        sentences = [info['sentence'] for info in sentence_info]
        
        # Create sentence vectors (simplified, using word presence)
        sentence_vectors = []
        for sentence in sentences:
            # Create a set of words for each sentence (excluding stopwords)
            words = set(word.lower() for word in sentence.split() 
                        if word.lower() not in STOPWORDS and len(word) > 3)
            sentence_vectors.append(words)
        
        # Initialize clusters with the first sentence
        clusters = [[sentence_info[0]]]
        used_indices = {0}
        
        # For each sentence, either add to an existing cluster or create a new one
        for i in range(1, len(sentence_info)):
            if i in used_indices:
                continue
                
            best_cluster = None
            best_similarity = 0
            
            # Find the most similar cluster
            for cluster_idx, cluster in enumerate(clusters):
                for info in cluster:
                    # Get index of the sentence in this cluster
                    j = sentences.index(info['sentence'])
                    
                    # Calculate similarity (Jaccard similarity of word sets)
                    if len(sentence_vectors[i]) == 0 or len(sentence_vectors[j]) == 0:
                        similarity = 0
                    else:
                        intersection = len(sentence_vectors[i] & sentence_vectors[j])
                        union = len(sentence_vectors[i] | sentence_vectors[j])
                        similarity = intersection / union
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_cluster = cluster_idx
            
            # Add to best cluster if similarity is above threshold
            if best_similarity > 0.2 and best_cluster is not None:
                clusters[best_cluster].append(sentence_info[i])
            else:
                # Create a new cluster
                clusters.append([sentence_info[i]])
            
            used_indices.add(i)
        
        return clusters
    
    def _generate_simple_summary(self, query: str, retrieved_docs: List[Tuple[str, str]]) -> str:
        """
        Generate a simple summary from the documents when more advanced methods fail.
        
        Args:
            query: User query
            retrieved_docs: List of (doc_name, content) tuples
            
        Returns:
            Simple summary text
        """
        summary = f"# Information about: {query}\n\n"
        
        for doc_name, content in retrieved_docs:
            # Extract first few sentences from each document
            sentences = nltk.sent_tokenize(content)
            
            # Get first paragraph (up to 5 sentences)
            first_paragraph = " ".join(sentences[:min(5, len(sentences))])
            
            # Format properly
            summary += f"## From {doc_name}:\n\n"
            summary += first_paragraph + "\n\n"
        
        return summary
    
    def generate_pdf_report(self, query: str, response: str, retrieved_docs: List[str]) -> str:
        """
        Generate a professionally formatted PDF report of the response.
        
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
            
            # Create a more professional PDF with ReportLab Platypus
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
            elements.append(Paragraph(f"Query Response: {query}", title_style))
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
    
    def _wrap_text(self, text: str, width: int) -> List[str]:
        """
        Wrap text to fit within specified width.
        
        Args:
            text: Text to wrap
            width: Maximum line width in characters
            
        Returns:
            List of wrapped lines
        """
        return textwrap.wrap(text, width=width)


# Main function to run the RAG system directly
def main():
    """Main function to run the RAG system."""
    
    # Configuration parameters - using your specified values
    KNOWLEDGE_BASE_PATH = "knowledge_base_docs"
    
    print(f"Initializing RAG system with:")
    print(f"- Knowledge base path: {KNOWLEDGE_BASE_PATH}")
    
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


if __name__ == "__main__":
    main()
