"""
Lightweight RAG Chatbot using Gemini API with sklearn embeddings
Install: pip install google-generativeai scikit-learn pypdf python-docx python-dotenv
Total size: ~50MB (vs 2GB+ with torch!)
"""

import google.generativeai as genai
import os
import pickle
from pathlib import Path
from dotenv import load_dotenv
import pypdf
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class LightweightRAGChatbot:
    def __init__(self, api_key):
        """Initialize RAG chatbot with Gemini API"""
        genai.configure(api_key=api_key)
        
        # Initialize Gemini model (only for generation)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Use sklearn TfidfVectorizer - lightweight and fast!
        print("🔧 Initializing TF-IDF vectorizer (lightweight)...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        # Storage
        self.documents = []
        self.metadatas = []
        self.vectors = None
        self.db_path = "./rag_db"
        
        # Create database directory
        os.makedirs(self.db_path, exist_ok=True)
        
        # Try to load existing database
        self._load_database()
        
        print("✓ RAG system initialized!")
    
    def _save_database(self):
        """Save documents and vectors to disk"""
        db_file = os.path.join(self.db_path, "database.pkl")
        with open(db_file, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadatas': self.metadatas,
                'vectorizer': self.vectorizer,
                'vectors': self.vectors
            }, f)
        print(f"💾 Database saved to {db_file}")
    
    def _load_database(self):
        """Load documents and vectors from disk"""
        db_file = os.path.join(self.db_path, "database.pkl")
        if os.path.exists(db_file):
            try:
                with open(db_file, 'rb') as f:
                    data = pickle.load(f)
                self.documents = data['documents']
                self.metadatas = data['metadatas']
                self.vectorizer = data['vectorizer']
                self.vectors = data['vectors']
                print(f"✓ Loaded {len(self.documents)} documents from existing database")
            except Exception as e:
                print(f"⚠ Could not load existing database: {e}")
    
    def add_documents(self, texts, metadatas=None):
        """Add documents to the database"""
        if not texts:
            return
            
        if metadatas is None:
            metadatas = [{"source": f"doc_{i}"} for i in range(len(texts))]
        
        # Add to storage
        self.documents.extend(texts)
        self.metadatas.extend(metadatas)
        
        # Recompute vectors for all documents
        print("🔄 Computing vectors...")
        self.vectors = self.vectorizer.fit_transform(self.documents)
        
        # Save to disk
        self._save_database()
        
        print(f"✓ Added {len(texts)} chunks (Total: {len(self.documents)})")
    
    def add_text_file(self, file_path):
        """Add content from a text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self._process_and_add_content(content, file_path)
        except Exception as e:
            print(f"✗ Error reading text file {file_path}: {e}")

    def add_pdf_file(self, file_path):
        """Add content from a PDF file"""
        try:
            reader = pypdf.PdfReader(file_path)
            content = ""
            for page in reader.pages:
                content += page.extract_text() + "\n"
            self._process_and_add_content(content, file_path)
        except Exception as e:
            print(f"✗ Error reading PDF file {file_path}: {e}")

    def add_docx_file(self, file_path):
        """Add content from a DOCX file"""
        try:
            doc = docx.Document(file_path)
            content = "\n".join([para.text for para in doc.paragraphs])
            self._process_and_add_content(content, file_path)
        except Exception as e:
            print(f"✗ Error reading DOCX file {file_path}: {e}")

    def _process_and_add_content(self, content, source):
        """Process and add content with chunking"""
        if not content.strip():
            print(f"⚠ Warning: Empty content in {source}")
            return
            
        chunks = self._chunk_text(content)
        if chunks:
            self.add_documents(chunks, [{"source": source}] * len(chunks))
            
    def load_documents_from_folder(self, folder_path):
        """Load all supported documents from a folder (recursive)"""
        print(f"\n📁 Scanning documents in '{folder_path}'...")
        supported_extensions = {
            '.txt': self.add_text_file,
            '.pdf': self.add_pdf_file,
            '.docx': self.add_docx_file
        }
        
        count = 0
        for root, _, files in os.walk(folder_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in supported_extensions:
                    file_path = os.path.join(root, file)
                    print(f"Processing: {file_path}")
                    supported_extensions[ext](file_path)
                    count += 1
        
        if count == 0:
            print(f"⚠ No supported documents found in '{folder_path}'")
            print(f"Supported formats: .txt, .pdf, .docx")
        else:
            print(f"\n✓ Successfully processed {count} file(s)")
    
    def _chunk_text(self, text, chunk_size=1000, overlap=200):
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        
        return chunks
    
    def query(self, question, n_results=3):
        """Query the RAG system"""
        try:
            # Check if we have any documents
            if len(self.documents) == 0:
                return "⚠ No documents in the knowledge base. Please add documents first."
            
            # Vectorize the question
            question_vector = self.vectorizer.transform([question])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(question_vector, self.vectors)[0]
            
            # Get top N results
            top_indices = np.argsort(similarities)[-n_results:][::-1]
            
            # Get relevant documents
            relevant_docs = [self.documents[i] for i in top_indices if similarities[i] > 0]
            
            if not relevant_docs:
                return "I don't have enough information to answer that question."
            
            # Build context from retrieved documents
            context = "\n\n".join(relevant_docs)
            
            # Create prompt with context
            prompt = f"""Based on the following context, please answer the question. If the answer is not in the context, say so.

Context:
{context}

Question: {question}

Answer:"""
            
            # Generate response using Gemini (only this uses API)
            response = self.model.generate_content(prompt)
            return response.text
        
        except Exception as e:
            return f"Error generating response: {e}"
    
    def clear_database(self):
        """Clear all documents from database"""
        self.documents = []
        self.metadatas = []
        self.vectors = None
        self._save_database()
        print("🗑️ Database cleared!")
    
    def chat(self):
        """Interactive chat loop"""
        print("\n" + "="*60)
        print("🤖 RAG Chatbot Ready!")
        print("="*60)
        print("Commands:")
        print("  - 'quit/exit/q' to exit")
        print("  - 'count' to see document count")
        print("  - 'clear' to clear database")
        print("-" * 60)
        
        while True:
            question = input("\n💬 You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if question.lower() == 'count':
                print(f"📊 Total chunks in database: {len(self.documents)}")
                continue
            
            if question.lower() == 'clear':
                confirm = input("⚠️  Clear all documents? (y/n): ").strip().lower()
                if confirm == 'y':
                    self.clear_database()
                continue
            
            if not question:
                continue
            
            try:
                answer = self.query(question)
                print(f"\n🤖 Bot: {answer}")
            except Exception as e:
                print(f"❌ Error: {e}")


# Example usage
if __name__ == "__main__":
    print("🚀 Starting Lightweight RAG Chatbot...")
    
    # Load environment variables
    load_dotenv()
    
    # Get API key from environment or input
    API_KEY = os.getenv("GEMINI_API_KEY")
    if not API_KEY:
        print("\n⚠ GEMINI_API_KEY not found in environment")
        API_KEY = input("Enter your Gemini API key: ").strip()
        
        if not API_KEY:
            print("❌ API key is required. Exiting...")
            exit(1)
    
    # Initialize chatbot
    print("\n🔧 Initializing RAG system...")
    chatbot = LightweightRAGChatbot(API_KEY)
    
    # Load documents from 'docs' folder
    docs_folder = "docs"
    if os.path.exists(docs_folder):
        # Check if database already exists
        if len(chatbot.documents) > 0:
            response = input(f"\n💡 Found existing database with {len(chatbot.documents)} chunks. Reload documents? (y/n): ").strip().lower()
            if response == 'y':
                chatbot.clear_database()
                chatbot.load_documents_from_folder(docs_folder)
        else:
            chatbot.load_documents_from_folder(docs_folder)
    else:
        print(f"\n📁 Folder '{docs_folder}' not found. Creating it...")
        os.makedirs(docs_folder)
        print(f"✓ Created '{docs_folder}' folder")
        print(f"\n💡 Tip: Place your .txt, .pdf, or .docx files in '{docs_folder}' and restart")
        
        # Check if user wants to continue without documents
        response = input("\nContinue without documents? (y/n): ").strip().lower()
        if response != 'y':
            print("👋 Exiting. Add documents and restart!")
            exit(0)

    # Start chat
    chatbot.chat()