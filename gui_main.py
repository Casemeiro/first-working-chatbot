"""
RAG Chatbot GUI using Tkinter (Desktop Application)
No additional dependencies needed - Tkinter comes with Python
"""

import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog
import google.generativeai as genai
import os
import pickle
from dotenv import load_dotenv
from main import LightweightRAGChatbot
import threading

class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 RAG Chatbot")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f0f0")
        
        self.chatbot = None
        self.api_key = None
        
        # Initialize UI
        self.setup_ui()
        self.initialize_chatbot()
    
    def setup_ui(self):
        """Create the GUI layout"""
        
        # Header Frame
        header_frame = tk.Frame(self.root, bg="#2c3e50")
        header_frame.pack(fill=tk.X, padx=0, pady=0)
        
        header_label = tk.Label(
            header_frame, 
            text="🤖 RAG Chatbot with Gemini AI",
            font=("Arial", 16, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        header_label.pack(pady=10)
        
        # Main Frame
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Chat Display Frame
        chat_label = tk.Label(main_frame, text="Chat History", font=("Arial", 10, "bold"), bg="#f0f0f0")
        chat_label.pack(anchor="w")
        
        self.chat_display = scrolledtext.ScrolledText(
            main_frame,
            wrap=tk.WORD,
            height=20,
            width=80,
            font=("Arial", 10),
            bg="white",
            state=tk.DISABLED
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Input Frame
        input_label = tk.Label(main_frame, text="Your Message", font=("Arial", 10, "bold"), bg="#f0f0f0")
        input_label.pack(anchor="w", pady=(10, 0))
        
        self.input_field = tk.Text(
            main_frame,
            height=3,
            width=80,
            font=("Arial", 10),
            bg="white"
        )
        self.input_field.pack(fill=tk.BOTH, pady=5)
        
        # Buttons Frame
        button_frame = tk.Frame(main_frame, bg="#f0f0f0")
        button_frame.pack(fill=tk.X, pady=10)
        
        send_btn = tk.Button(
            button_frame,
            text="📤 Send",
            command=self.send_message,
            font=("Arial", 11, "bold"),
            bg="#27ae60",
            fg="white",
            padx=20,
            pady=10
        )
        send_btn.pack(side=tk.LEFT, padx=5)
        
        load_btn = tk.Button(
            button_frame,
            text="📁 Load Documents",
            command=self.load_documents,
            font=("Arial", 11, "bold"),
            bg="#3498db",
            fg="white",
            padx=20,
            pady=10
        )
        load_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = tk.Button(
            button_frame,
            text="🗑️ Clear DB",
            command=self.clear_database,
            font=("Arial", 11, "bold"),
            bg="#e74c3c",
            fg="white",
            padx=20,
            pady=10
        )
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Status Frame
        status_frame = tk.Frame(self.root, bg="#ecf0f1", relief=tk.SUNKEN, height=30)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = tk.Label(
            status_frame,
            text="Initializing...",
            font=("Arial", 9),
            bg="#ecf0f1",
            fg="#2c3e50"
        )
        self.status_label.pack(anchor="w", padx=10, pady=5)
        
        # Bind Enter key for sending messages
        self.input_field.bind("<Control-Return>", lambda e: self.send_message())
    
    def initialize_chatbot(self):
        """Initialize chatbot in background thread"""
        thread = threading.Thread(target=self._init_chatbot_thread)
        thread.daemon = True
        thread.start()
    
    def _init_chatbot_thread(self):
        """Background initialization"""
        try:
            load_dotenv()
            self.api_key = os.getenv("GEMINI_API_KEY")
            
            if not self.api_key:
                self.root.after(0, lambda: messagebox.showerror(
                    "API Key Missing",
                    "GEMINI_API_KEY not found in .env file\n\nPlease set it and restart."
                ))
                return
            
            self.chatbot = LightweightRAGChatbot(self.api_key)
            
            docs_folder = "docs"
            if os.path.exists(docs_folder) and len(self.chatbot.documents) == 0:
                self.chatbot.load_documents_from_folder(docs_folder)
            
            self.root.after(0, self.on_ready)
        
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to initialize: {e}"))
    
    def on_ready(self):
        """Called when chatbot is ready"""
        self.update_status(f"✓ Ready! Database: {len(self.chatbot.documents)} chunks")
        self.add_to_chat("🤖 Bot", "Hello! I'm your RAG Chatbot. Ask me anything about your documents!")
    
    def update_status(self, message):
        """Update status bar"""
        self.status_label.config(text=message)
    
    def add_to_chat(self, sender, message):
        """Add message to chat display"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"\n{sender}: {message}\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def send_message(self):
        """Send message and get response"""
        if not self.chatbot:
            messagebox.showwarning("Not Ready", "Chatbot is still initializing...")
            return
        
        question = self.input_field.get("1.0", tk.END).strip()
        if not question:
            messagebox.showwarning("Empty Message", "Please enter a message!")
            return
        
        # Clear input
        self.input_field.delete("1.0", tk.END)
        
        # Add user message
        self.add_to_chat("💬 You", question)
        self.update_status("⏳ Processing...")
        
        # Get response in background
        thread = threading.Thread(target=self._get_response_thread, args=(question,))
        thread.daemon = True
        thread.start()
    
    def _get_response_thread(self, question):
        """Get response in background thread"""
        try:
            answer = self.chatbot.query(question)
            self.root.after(0, lambda: self._display_response(answer))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to get response: {e}"))
            self.root.after(0, lambda: self.update_status("❌ Error"))
    
    def _display_response(self, answer):
        """Display bot response"""
        self.add_to_chat("🤖 Bot", answer)
        doc_count = len(self.chatbot.documents)
        self.update_status(f"✓ Ready! Database: {doc_count} chunks")
    
    def load_documents(self):
        """Let user select documents to load"""
        if not self.chatbot:
            messagebox.showwarning("Not Ready", "Chatbot is still initializing...")
            return
        
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Text files", "*.txt"),
                ("PDF files", "*.pdf"),
                ("Word documents", "*.docx"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.update_status(f"⏳ Loading {os.path.basename(file_path)}...")
            
            thread = threading.Thread(target=self._load_file_thread, args=(file_path,))
            thread.daemon = True
            thread.start()
    
    def _load_file_thread(self, file_path):
        """Load file in background"""
        try:
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext == '.txt':
                self.chatbot.add_text_file(file_path)
            elif ext == '.pdf':
                self.chatbot.add_pdf(file_path)
            elif ext == '.docx':
                self.chatbot.add_docx(file_path)
            
            self.root.after(0, lambda: messagebox.showinfo(
                "Success",
                f"Document loaded!\nTotal chunks: {len(self.chatbot.documents)}"
            ))
            self.root.after(0, lambda: self.update_status(f"✓ Ready! Database: {len(self.chatbot.documents)} chunks"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load file: {e}"))
            self.root.after(0, lambda: self.update_status("❌ Error loading file"))
    
    def clear_database(self):
        """Clear database with confirmation"""
        if messagebox.askyesno("Clear Database", "Are you sure you want to clear all documents?"):
            self.chatbot.clear_database()
            self.add_to_chat("🤖 Bot", "Database cleared!")
            self.update_status("✓ Ready! Database: 0 chunks")


if __name__ == "__main__":
    root = tk.Tk()
    gui = ChatbotGUI(root)
    root.mainloop()
