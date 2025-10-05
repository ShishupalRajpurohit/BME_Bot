"""
Vector store management module
Handles FAISS vector store operations with proper error handling
"""
from pathlib import Path
from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from utils.logger import setup_logger

logger = setup_logger(__name__)

class VectorStoreManager:
    """Manages FAISS vector store operations"""
    
    def __init__(self, embedding_model_name: str, vectorstore_path: Path):
        """
        Initialize the vector store manager
        
        Args:
            embedding_model_name: HuggingFace embedding model name
            vectorstore_path: Path to FAISS vector store directory
        """
        self.embedding_model_name = embedding_model_name
        self.vectorstore_path = vectorstore_path
        self._embedding_model = None
        self._vectorstore = None
        
    @property
    def embedding_model(self) -> HuggingFaceEmbeddings:
        """Lazy load embedding model"""
        if self._embedding_model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            try:
                self._embedding_model = HuggingFaceEmbeddings(
                    model_name=self.embedding_model_name
                )
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {str(e)}")
                raise
        return self._embedding_model
    
    def load_vectorstore(self) -> FAISS:
        """
        Load FAISS vector store from disk
        
        Returns:
            Loaded FAISS vector store
            
        Raises:
            FileNotFoundError: If vector store doesn't exist
            Exception: For other loading errors
        """
        if self._vectorstore is not None:
            return self._vectorstore
            
        if not self.vectorstore_path.exists():
            error_msg = f"Vector store not found at {self.vectorstore_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            logger.info(f"Loading vector store from {self.vectorstore_path}")
            try:
                # Fixed: Use self.vectorstore_path and self.embedding_model
                self._vectorstore = FAISS.load_local(
                    str(self.vectorstore_path),
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
            except TypeError:
                # Backward compatibility for older FAISS versions
                self._vectorstore = FAISS.load_local(
                    str(self.vectorstore_path),
                    self.embedding_model
                )
            logger.info("Vector store loaded successfully")
            return self._vectorstore
        except Exception as e:
            logger.error(f"Failed to load vector store: {str(e)}")
            raise
    
    def create_vectorstore(self, documents: list) -> FAISS:
        """
        Create new FAISS vector store from documents
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            Created FAISS vector store
        """
        if not documents:
            raise ValueError("Cannot create vector store from empty document list")
        
        try:
            logger.info(f"Creating vector store with {len(documents)} documents")
            self._vectorstore = FAISS.from_documents(
                documents,
                self.embedding_model
            )
            logger.info("Vector store created successfully")
            return self._vectorstore
        except Exception as e:
            logger.error(f"Failed to create vector store: {str(e)}")
            raise
    
    def save_vectorstore(self) -> None:
        """Save vector store to disk"""
        if self._vectorstore is None:
            raise ValueError("No vector store to save. Load or create one first.")
        
        try:
            self.vectorstore_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving vector store to {self.vectorstore_path}")
            self._vectorstore.save_local(str(self.vectorstore_path))
            logger.info("Vector store saved successfully")
        except Exception as e:
            logger.error(f"Failed to save vector store: {str(e)}")
            raise
    
    def get_retriever(self, k: int = 3):
        """
        Get retriever from vector store
        
        Args:
            k: Number of documents to retrieve
            
        Returns:
            LangChain retriever instance
        """
        if self._vectorstore is None:
            self.load_vectorstore()
        
        logger.debug(f"Creating retriever with k={k}")
        return self._vectorstore.as_retriever(search_kwargs={'k': k})