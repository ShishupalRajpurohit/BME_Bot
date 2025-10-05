"""
Document processing module
Handles PDF loading, text extraction, and chunking
"""
import os
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from utils.logger import setup_logger

logger = setup_logger(__name__)

class DocumentProcessor:
    """Processes documents for vector store ingestion"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize document processor
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_pdf_files(self, data_path: Path) -> List[Document]:
        """
        Load all PDF files from directory with OCR fallback
        
        Args:
            data_path: Path to directory containing PDFs
            
        Returns:
            List of loaded Document objects
        """
        if not data_path.exists():
            logger.error(f"Data path does not exist: {data_path}")
            raise FileNotFoundError(f"Directory not found: {data_path}")
        
        documents = []
        pdf_files = list(data_path.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {data_path}")
            return documents
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"Processing: {pdf_file.name}")
                docs = self._load_single_pdf(pdf_file)
                documents.extend(docs)
                logger.info(f"✓ Loaded {len(docs)} pages from {pdf_file.name}")
            except Exception as e:
                logger.error(f"✗ Failed to process {pdf_file.name}: {str(e)}")
        
        logger.info(f"Total pages loaded: {len(documents)}")
        return documents
    
    def _load_single_pdf(self, file_path: Path) -> List[Document]:
        """
        Load a single PDF with OCR fallback
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of Document objects
        """
        try:
            # Try standard text extraction first
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()
            
            # Check if text was extracted
            if docs and any(len(doc.page_content.strip()) > 50 for doc in docs):
                return docs
            
            # Fallback to OCR if no meaningful text
            logger.warning(f"Low text content in {file_path.name}, using OCR")
            return self._load_with_ocr(file_path)
            
        except Exception as e:
            logger.warning(f"Standard loading failed for {file_path.name}: {str(e)}")
            return self._load_with_ocr(file_path)
    
    def _load_with_ocr(self, file_path: Path) -> List[Document]:
        """
        Load PDF using OCR
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of Document objects
        """
        try:
            loader = UnstructuredPDFLoader(
                str(file_path),
                strategy="ocr_only"
            )
            return loader.load()
        except Exception as e:
            logger.error(f"OCR loading failed for {file_path.name}: {str(e)}")
            return []
    
    def create_chunks(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        if not documents:
            logger.warning("No documents to chunk")
            return []
        
        logger.info(f"Splitting {len(documents)} documents into chunks")
        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} text chunks")
            return chunks
        except Exception as e:
            logger.error(f"Chunking failed: {str(e)}")
            raise
    
    def process_directory(self, data_path: Path) -> List[Document]:
        """
        Complete pipeline: load PDFs and create chunks
        
        Args:
            data_path: Path to directory containing PDFs
            
        Returns:
            List of chunked Document objects ready for indexing
        """
        logger.info("Starting document processing pipeline")
        documents = self.load_pdf_files(data_path)
        chunks = self.create_chunks(documents)
        logger.info("Document processing completed")
        return chunks