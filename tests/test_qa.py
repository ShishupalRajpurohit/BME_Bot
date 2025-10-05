"""
Unit tests for BME Bot
Run with: pytest tests/
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from core.vectorstore import VectorStoreManager
from core.qa_chain import QAChainManager
from core.document_processor import DocumentProcessor

class TestVectorStoreManager:
    """Test vector store operations"""
    
    def test_initialization(self):
        """Test vector store manager initialization"""
        manager = VectorStoreManager(
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            vectorstore_path=Path("test_vectorstore")
        )
        assert manager.embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert manager.vectorstore_path == Path("test_vectorstore")
    
    def test_embedding_model_lazy_loading(self):
        """Test that embedding model is lazy loaded"""
        manager = VectorStoreManager(
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            vectorstore_path=Path("test_vectorstore")
        )
        assert manager._embedding_model is None
        # Access property to trigger loading
        model = manager.embedding_model
        assert model is not None
    
    def test_load_nonexistent_vectorstore(self):
        """Test loading vector store that doesn't exist"""
        manager = VectorStoreManager(
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            vectorstore_path=Path("nonexistent_path")
        )
        with pytest.raises(FileNotFoundError):
            manager.load_vectorstore()

class TestQAChainManager:
    """Test QA chain operations"""
    
    def test_initialization(self):
        """Test QA chain manager initialization"""
        manager = QAChainManager(
            groq_api_key="test_key",
            model_name="test_model",
            temperature=0.5
        )
        assert manager.groq_api_key == "test_key"
        assert manager.model_name == "test_model"
        assert manager.temperature == 0.5
    
    def test_query_without_chain(self):
        """Test querying without creating chain first"""
        manager = QAChainManager(
            groq_api_key="test_key",
            model_name="test_model"
        )
        with pytest.raises(ValueError, match="QA chain not initialized"):
            manager.query("test question")
    
    def test_format_response_basic(self):
        """Test basic response formatting"""
        manager = QAChainManager(
            groq_api_key="test_key",
            model_name="test_model"
        )
        response = {"result": "Test answer"}
        formatted = manager.format_response(response, include_sources=False)
        assert formatted == "Test answer"
    
    def test_format_response_with_sources(self):
        """Test response formatting with sources"""
        manager = QAChainManager(
            groq_api_key="test_key",
            model_name="test_model"
        )
        
        mock_doc = Mock()
        mock_doc.metadata = {"source": "test.pdf", "page": 1}
        
        response = {
            "result": "Test answer",
            "source_documents": [mock_doc]
        }
        formatted = manager.format_response(response, include_sources=True)
        assert "Test answer" in formatted
        assert "Sources:" in formatted
        assert "test.pdf" in formatted

class TestDocumentProcessor:
    """Test document processing"""
    
    def test_initialization(self):
        """Test document processor initialization"""
        processor = DocumentProcessor(chunk_size=1000, chunk_overlap=100)
        assert processor.chunk_size == 1000
        assert processor.chunk_overlap == 100
    
    def test_load_nonexistent_directory(self):
        """Test loading from non-existent directory"""
        processor = DocumentProcessor()
        with pytest.raises(FileNotFoundError):
            processor.load_pdf_files(Path("nonexistent_dir"))
    
    def test_create_chunks_empty_list(self):
        """Test chunking empty document list"""
        processor = DocumentProcessor()
        chunks = processor.create_chunks([])
        assert chunks == []
    
    @patch('core.document_processor.PyPDFLoader')
    def test_load_single_pdf_success(self, mock_loader):
        """Test successful PDF loading"""
        # Mock document with content
        mock_doc = Mock()
        mock_doc.page_content = "A" * 100  # Sufficient content
        
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = [mock_doc]
        mock_loader.return_value = mock_loader_instance
        
        processor = DocumentProcessor()
        docs = processor._load_single_pdf(Path("test.pdf"))
        
        assert len(docs) == 1
        assert len(docs[0].page_content) == 100

class TestConfig:
    """Test configuration"""
    
    def test_config_values(self):
        """Test that config has required values"""
        assert hasattr(Config, 'BASE_DIR')
        assert hasattr(Config, 'DATA_DIR')
        assert hasattr(Config, 'VECTORSTORE_DIR')
        assert hasattr(Config, 'EMBEDDING_MODEL')
        assert hasattr(Config, 'LLM_MODEL')
    
    def test_directories_are_paths(self):
        """Test that directory configs are Path objects"""
        assert isinstance(Config.BASE_DIR, Path)
        assert isinstance(Config.DATA_DIR, Path)
        assert isinstance(Config.VECTORSTORE_DIR, Path)
    
    def test_model_names_are_strings(self):
        """Test that model names are strings"""
        assert isinstance(Config.EMBEDDING_MODEL, str)
        assert isinstance(Config.LLM_MODEL, str)

# Integration test (requires actual API key and vector store)
@pytest.mark.integration
class TestIntegration:
    """Integration tests - run only when needed"""
    
    @pytest.mark.skipif(
        not Config.GROQ_API_KEY,
        reason="GROQ_API_KEY not set"
    )
    def test_end_to_end_query(self):
        """Test complete query pipeline"""
        # This test requires:
        # 1. Valid GROQ_API_KEY
        # 2. Existing vector store
        pytest.skip("Integration test - run manually")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])