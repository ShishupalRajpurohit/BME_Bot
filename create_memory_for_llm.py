from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, UnstructuredPDFLoader  # Import PDF loader to read PDFs from a directory
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Import text splitter to break text into chunks
from langchain_huggingface import HuggingFaceEmbeddings  # Import HuggingFace embedding model utility
from langchain_community.vectorstores import FAISS  # Import FAISS vector store for storing embeddings
import os

# Step 1: load raw pdfs
Data_Path = "data/"  # Path to folder containing PDF files

def load_pdf_files(data):
    documents = []

    # Get all PDF file paths
    pdf_files = [os.path.join(data, f) for f in os.listdir(data) if f.endswith(".pdf")]

    for file_path in pdf_files:
        try:
            # Try normal text extraction
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            # If text successfully extracted, add it
            if docs and any(len(doc.page_content.strip()) > 0 for doc in docs):
                documents.extend(docs)
            else:
                # If no text, fallback to OCR
                print(f"[OCR] Falling back to OCR for: {file_path}")
                ocr_loader = UnstructuredPDFLoader(file_path, strategy="ocr_only")
                ocr_docs = ocr_loader.load()
                documents.extend(ocr_docs)

        except Exception as e:
            print(f"[ERROR] Could not load {file_path} with PyPDFLoader. Using OCR. Error: {e}")
            ocr_loader = UnstructuredPDFLoader(file_path, strategy="ocr_only")
            ocr_docs = ocr_loader.load()
            documents.extend(ocr_docs)

    return documents

documents = load_pdf_files(data=Data_Path)  # Load PDF documents
print("length of pdf pages: ", len(documents))  # Print number of pages extracted

# Step 2: create chunks

def create_chunks(extract_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)  # Split text into chunks of 500 chars with 50 overlap
    text_chunks = text_splitter.split_documents(extract_data)  # Perform the split
    return text_chunks

text_chunks = create_chunks(extract_data=documents)  # Create chunks from documents
print("Length of text Chunks:", len(text_chunks))  # Print number of text chunks

# Step 3: create vector embeddings

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Load sentence embedding model
    return embedding_model

embedding_model = get_embedding_model()  # Initialize embedding model

# Step 4: store embeddings to FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"  # Path to save FAISS vector store
db = FAISS.from_documents(text_chunks, embedding_model)  # Create FAISS vector store from text chunks and embeddings
db.save_local(DB_FAISS_PATH)  # Save vector store locally
