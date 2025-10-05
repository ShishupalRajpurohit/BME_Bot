"""
Build Vector Store Script
Processes PDF documents and creates FAISS vector store
Run this script before starting the Streamlit app
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from core.document_processor import DocumentProcessor
from core.vectorstore import VectorStoreManager
from utils.logger import setup_logger

logger = setup_logger(__name__, log_dir=Config.LOGS_DIR)

def main():
    """Main indexing pipeline"""
    
    print("=" * 60)
    print("BME Bot - Vector Store Builder")
    print("=" * 60)
    
    try:
        # Step 1: Check if data directory exists
        if not Config.DATA_DIR.exists():
            logger.error(f"Data directory not found: {Config.DATA_DIR}")
            print(f"❌ Error: Data directory not found at {Config.DATA_DIR}")
            print("💡 Please create the 'data' directory and add PDF files")
            return
        
        print(f"\n📂 Data directory: {Config.DATA_DIR}")
        
        # Step 2: Initialize document processor
        print("\n🔧 Initializing document processor...")
        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
        # Step 3: Process documents
        print("\n📄 Processing PDF documents...")
        chunks = doc_processor.process_directory(Config.DATA_DIR)
        
        if not chunks:
            logger.error("No documents were processed")
            print("❌ No documents found or processed")
            print("💡 Ensure PDF files are in the data directory")
            return
        
        print(f"✅ Created {len(chunks)} text chunks")
        
        # Step 4: Initialize vector store manager
        print("\n🧠 Initializing vector store manager...")
        vectorstore_manager = VectorStoreManager(
            embedding_model_name=Config.EMBEDDING_MODEL,
            vectorstore_path=Config.VECTORSTORE_DIR
        )
        
        # Step 5: Create vector store
        print("\n🔨 Creating FAISS vector store...")
        vectorstore_manager.create_vectorstore(chunks)
        
        # Step 6: Save vector store
        print(f"\n💾 Saving vector store to {Config.VECTORSTORE_DIR}...")
        vectorstore_manager.save_vectorstore()
        
        print("\n" + "=" * 60)
        print("✅ Vector store built successfully!")
        print("=" * 60)
        print(f"\n📊 Summary:")
        print(f"   - Total chunks: {len(chunks)}")
        print(f"   - Embedding model: {Config.EMBEDDING_MODEL}")
        print(f"   - Saved to: {Config.VECTORSTORE_DIR}")
        print(f"\n🚀 You can now run: streamlit run app.py")
        
    except Exception as e:
        logger.error(f"Vector store building failed: {str(e)}", exc_info=True)
        print(f"\n❌ Error: {str(e)}")
        print("💡 Check logs for detailed error information")
        sys.exit(1)

if __name__ == "__main__":
    main()