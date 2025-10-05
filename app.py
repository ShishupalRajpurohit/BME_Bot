"""
BME Bot - Biomedical Engineering Assistant
Production-ready Streamlit application with proper error handling and logging
"""
import streamlit as st
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from core.vectorstore import VectorStoreManager
from core.qa_chain import QAChainManager
from utils.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__, log_dir=Config.LOGS_DIR)

# Page configuration
st.set_page_config(
    page_title=Config.PAGE_TITLE,
    page_icon=Config.PAGE_ICON,
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .main-header {
        text-align: center;
        padding: 1rem 0;
        color: #1f77b4;
    }
    .status-box {
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_system():
    """
    Initialize vector store and QA chain
    Cached to prevent reloading on every interaction
    """
    try:
        logger.info("Initializing BME Bot system")
        
        # Validate API key
        if not Config.GROQ_API_KEY:
            error_msg = "GROQ_API_KEY not found in environment variables"
            logger.error(error_msg)
            st.error(f"❌ Configuration Error: {error_msg}")
            st.stop()
        
        # Initialize vector store manager
        vectorstore_manager = VectorStoreManager(
            embedding_model_name=Config.EMBEDDING_MODEL,
            vectorstore_path=Config.VECTORSTORE_DIR
        )
        
        # Load vector store
        vectorstore_manager.load_vectorstore()
        retriever = vectorstore_manager.get_retriever(k=Config.RETRIEVAL_K)
        
        # Initialize QA chain manager
        qa_manager = QAChainManager(
            groq_api_key=Config.GROQ_API_KEY,
            model_name=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE
        )
        
        # Create QA chain
        qa_manager.create_qa_chain(
            retriever=retriever,
            prompt_template=Config.CUSTOM_PROMPT_TEMPLATE,
            return_source_documents=True
        )
        
        logger.info("System initialization successful")
        return qa_manager, True
        
    except FileNotFoundError as e:
        logger.error(f"Vector store not found: {str(e)}")
        st.error("❌ Vector store not found. Please run the indexing script first.")
        st.info("💡 Run: `python scripts/build_vectorstore.py`")
        return None, False
        
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        st.error(f"❌ System initialization failed: {str(e)}")
        return None, False

def display_header():
    """Display app header with branding"""
    st.markdown(f"<h1 class='main-header'>{Config.PAGE_ICON} BME Bot</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Your Intelligent Biomedical Engineering Assistant</p>", unsafe_allow_html=True)
    st.divider()

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        logger.info("New chat session started")
    
    if 'total_queries' not in st.session_state:
        st.session_state.total_queries = 0

def display_chat_history():
    """Display all previous messages"""
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

def process_user_query(qa_manager: QAChainManager, user_input: str):
    """
    Process user query and generate response
    
    Args:
        qa_manager: QA chain manager instance
        user_input: User's input text
    """
    # Display user message
    with st.chat_message('user'):
        st.markdown(user_input)
    
    # Add to history
    st.session_state.messages.append({'role': 'user', 'content': user_input})
    st.session_state.total_queries += 1
    
    # Generate response with spinner
    with st.chat_message('assistant'):
        with st.spinner('Thinking...'):
            try:
                logger.info(f"Processing query #{st.session_state.total_queries}")
                
                # Get response from QA chain
                response = qa_manager.query(user_input)
                
                # Format response
                result = qa_manager.format_response(
                    response,
                    include_sources=False  # Set to True if you want to show sources
                )
                
                # Display response
                st.markdown(result)
                
                # Add to history
                st.session_state.messages.append({'role': 'assistant', 'content': result})
                
                logger.info("Query processed successfully")
                
            except Exception as e:
                error_msg = f"I encountered an error: {str(e)}"
                logger.error(f"Query processing error: {str(e)}")
                st.error(error_msg)
                st.session_state.messages.append({'role': 'assistant', 'content': error_msg})

def display_sidebar(qa_manager):
    """Display sidebar with info and controls"""
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **BME Bot** is an AI-powered assistant specialized in biomedical engineering topics.
        
        **Features:**
        - 📚 RAG-based answers from technical documents
        - 💬 Natural conversation support
        - 🔍 Context-aware responses
        
        **Tech Stack:**
        - LangChain + FAISS
        - Groq LLM (DeepSeek)
        - Streamlit
        """)
        
        st.divider()
        
        # Stats
        st.header("📊 Session Stats")
        st.metric("Total Queries", st.session_state.total_queries)
        st.metric("Messages", len(st.session_state.messages))
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.total_queries = 0
            logger.info("Chat history cleared")
            st.rerun()
        
        st.divider()
        
        # System status
        st.header("⚙️ System Status")
        st.success("✅ Vector Store: Loaded")
        st.success("✅ LLM: Connected")
        st.info(f"🤖 Model: {Config.LLM_MODEL}")

def main():
    """Main application function"""
    
    # Display header
    display_header()
    
    # Initialize session state
    initialize_session_state()
    
    # Initialize system (cached)
    qa_manager, initialized = initialize_system()
    
    if not initialized:
        st.stop()
    
    # Display sidebar
    display_sidebar(qa_manager)
    
    # Display chat history
    display_chat_history()
    
    # Show initial message if first visit
    if len(st.session_state.messages) == 0:
        with st.chat_message('assistant'):
            st.markdown(Config.INITIAL_MESSAGE)
        st.session_state.messages.append({
            'role': 'assistant',
            'content': Config.INITIAL_MESSAGE
        })
    
    # Chat input
    if user_input := st.chat_input("Ask me anything about biomedical engineering..."):
        process_user_query(qa_manager, user_input)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Application crashed: {str(e)}", exc_info=True)
        st.error("🚨 Critical error occurred. Please check logs.")
        st.exception(e)