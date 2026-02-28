"""
RAG Chatbot GUI using Streamlit (Web Application)
Run with: streamlit run app_streamlit.py
"""

import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from main import LightweightRAGChatbot
import tempfile

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="🤖 RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #000000;
    }
    .stChatMessage {
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
    st.session_state.api_key = None
    st.session_state.chat_history = []
    st.session_state.initialized = False

def initialize_chatbot():
    """Initialize the chatbot with API key"""
    api_key = st.session_state.api_key
    
    if not api_key:
        st.warning("⚠️ Please enter your Gemini API key in the sidebar")
        return False
    
    try:
        st.session_state.chatbot = LightweightRAGChatbot(api_key)
        st.session_state.initialized = True
        return True
    except Exception as e:
        st.error(f"❌ Error initializing chatbot: {e}")
        return False

def load_documents_from_files(uploaded_files):
    """Load documents from uploaded files"""
    if not uploaded_files or not st.session_state.chatbot:
        return False
    
    try:
        for uploaded_file in uploaded_files:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_path = tmp_file.name
            
            try:
                # Load based on file type
                file_ext = Path(uploaded_file.name).suffix.lower()
                
                if file_ext == '.pdf':
                    st.session_state.chatbot.add_pdf_file(tmp_path)
                elif file_ext == '.docx':
                    st.session_state.chatbot.add_docx_file(tmp_path)
                elif file_ext == '.txt':
                    st.session_state.chatbot.add_text_file(tmp_path)
                else:
                    st.warning(f"⚠️ Unsupported file format: {file_ext}")
                    continue
                
                st.success(f"✓ Successfully loaded: {uploaded_file.name}")
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)
        
        return True
    except Exception as e:
        st.error(f"❌ Error loading documents: {e}")
        return False

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key Input
    api_key_input = st.text_input(
        "🔑 Gemini API Key",
        value=st.session_state.api_key or "",
        type="password",
        help="Get your API key from https://ai.google.dev"
    )
    
    if api_key_input:
        st.session_state.api_key = api_key_input
    
    # Initialize Button
    if st.button("🚀 Initialize Chatbot", use_container_width=True):
        if initialize_chatbot():
            st.success("✓ Chatbot initialized!")
        else:
            st.session_state.initialized = False
    
    st.divider()
    
    # Document Management Section
    st.subheader("📁 Document Management")
    
    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload documents (.txt, .pdf, .docx)",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        help="Upload documents to build your knowledge base"
    )
    
    if uploaded_files:
        if st.button("📤 Load Documents", use_container_width=True):
            if st.session_state.chatbot:
                if load_documents_from_files(uploaded_files):
                    st.rerun()
            else:
                st.error("❌ Please initialize the chatbot first")
    
    # Load from folder
    if st.button("📂 Load from 'docs' folder", use_container_width=True):
        if st.session_state.chatbot:
            docs_folder = "docs"
            if os.path.exists(docs_folder):
                try:
                    st.session_state.chatbot.load_documents_from_folder(docs_folder)
                    st.success(f"✓ Loaded documents from '{docs_folder}'")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")
            else:
                st.warning(f"📁 '{docs_folder}' folder not found")
        else:
            st.error("❌ Please initialize the chatbot first")
    
    st.divider()
    
    # Stats Section
    st.subheader("📊 Database Stats")
    
    if st.session_state.chatbot:
        doc_count = len(st.session_state.chatbot.documents)
        st.metric("📚 Documents in Database", doc_count)
        
        if st.button("🗑️ Clear Database", use_container_width=True):
            if st.confirm_button("Are you sure? This will remove all documents."):
                st.session_state.chatbot.clear_database()
                st.success("✓ Database cleared!")
                st.rerun()
    else:
        st.info("Initialize chatbot to see stats")

# Main Chat Interface
st.header("🤖 RAG Chatbot")

if not st.session_state.initialized:
    st.info("👈 Configure your API key in the sidebar to get started")
else:
    # Chat display area
    st.subheader("💬 Chat History")
    
    # Display previous messages
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                st.chat_message("assistant").write(msg["content"])
    
    st.divider()
    
    # User input
    user_input = st.chat_input("💬 Ask your question...")
    
    if user_input:
        # Display user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.write(user_input)
        
        # Generate response
        with st.chat_message("assistant"):
            if len(st.session_state.chatbot.documents) == 0:
                response = "⚠️ No documents in the knowledge base. Please add documents first."
            else:
                with st.spinner("🤔 Thinking..."):
                    response = st.session_state.chatbot.query(user_input)
            
            st.write(response)
        
        # Add to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.8em;'>
    🤖 RAG Chatbot powered by Google Gemini API
    </div>
""", unsafe_allow_html=True)
