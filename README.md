# 🔬 BME Bot - Biomedical Engineering AI Assistant

[![Python 3.11+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50-FF4B4B.svg)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-3.27-green.svg)](https://langchain.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)

A production-ready, RAG-powered chatbot specialized in biomedical engineering topics. Built with LangChain, FAISS, and Groq LLM for intelligent document-based question answering.

## 🌟 Features

- **🧠 RAG Architecture**: Retrieval-Augmented Generation for accurate, context-based responses
- **📚 Document Processing**: Automatic PDF processing with OCR fallback
- **💬 Dual Mode Operation**: Technical Q&A + casual conversation support
- **🚀 Production Ready**: Proper error handling, logging, and monitoring
- **🐳 Docker Support**: One-command deployment with Docker Compose
- **✅ Tested**: Unit tests with pytest
- **📊 Session Management**: Track queries and conversation history

## 🏗️ Architecture

```

┌─────────────┐
│   User UI   │  (Streamlit)
└──────┬──────┘
│
┌──────▼──────────────┐
│   QA Chain Manager  │
│  (LangChain + Groq) │
└──────┬──────────────┘
│
┌──────▼──────────────┐
│ Vector Store (FAISS)│
│   + Embeddings      │
└──────┬──────────────┘
│
┌──────▼──────────────┐
│  PDF Documents      │
│  (Technical Docs)   │
└─────────────────────┘

```

## 📁 Project Structure

```

BME_Bot/
├── app.py                      # Main Streamlit application
├── config.py                   # Centralized configuration
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker container setup
├── docker-compose.yml          # Docker Compose configuration
│
├── core/                       # Core functionality modules
│   ├── **init**.py
│   ├── vectorstore.py         # FAISS vector store management
│   ├── qa_chain.py            # QA chain operations
│   └── document_processor.py   # PDF processing & chunking
│
├── utils/                      # Utility modules
│   ├── **init**.py
│   └── logger.py              # Logging configuration
│
├── scripts/                    # Utility scripts
│   ├── **init**.py
│   └── build_vectorstore.py   # Index documents script
│
├── tests/                      # Unit tests
│   ├── **init**.py
│   └── test_qa.py             # Test suite
│
├── data/                       # PDF documents (add your files here)
├── vectorstore/                # FAISS index storage
│   └── db_faiss/
├── logs/                       # Application logs
│
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules
├── LICENSE.txt                # MIT License
└── README.md                  # This file

````

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- Groq API Key ([Get one free](https://console.groq.com))
- PDF documents for your knowledge base

### Installation

1. **Clone the repository**
```bash
gh repo clone https://github.com/ShishupalRajpurohit/BME_Bot
cd BME_Bot
````

2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

5. **Add your PDF documents**

```bash
# Place PDF files in the data/ directory
cp your_documents.pdf data/
```

6. **Build vector store**

```bash
python scripts/build_vectorstore.py
```

7. **Run the application**

```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser!

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Using Docker directly

```bash
# Build image
docker build -t bme-bot .

# Run container
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/vectorstore:/app/vectorstore \
  -e GROQ_API_KEY=your_key_here \
  bme-bot
```

## ☁️ Deploy to Streamlit Cloud

*(Link/details to be added later)*

## 🧪 Testing

Run tests with pytest:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=core --cov=utils

# Run specific test file
pytest tests/test_qa.py -v
```

## 📝 Configuration

Edit `config.py` to customize:

```python
# Model settings
LLM_MODEL = "deepseek-r1-distill-llama-70b"  # Change model
LLM_TEMPERATURE = 0.0                         # Adjust creativity

# Retrieval settings
RETRIEVAL_K = 6                               # Number of docs to retrieve
CHUNK_SIZE = 500                              # Text chunk size
CHUNK_OVERLAP = 50                            # Chunk overlap
```

## 🎯 Usage Examples

### Technical Q&A

```
User: What are the key components of an ECG machine?
Bot: Based on the documentation, an ECG machine consists of...
```

### Casual Chat

```
User: Hello! How are you?
Bot: Hello! I'm doing great, thank you for asking! How can I help you today?
```

### Troubleshooting

```
User: How do I calibrate a dialysis machine?
Bot: According to the manual, calibration involves these steps:
1. ...
2. ...
```

## 🔧 Troubleshooting

### Vector Store Not Found

```bash
# Rebuild the vector store
python scripts/build_vectorstore.py
```

### OCR Issues

```bash
# Install Tesseract OCR
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

### API Key Errors

* Verify your `.env` file exists
* Check that `GROQ_API_KEY` is set correctly
* Ensure no quotes around the key value

## 📊 Performance

* **Response Time**: ~2-4 seconds per query
* **Accuracy**: Depends on document quality
* **Uptime**: 99%+ with proper deployment
* **Concurrent Users**: Supports multiple users (Streamlit limitation)

## 🛠️ Technology Stack

| Component               | Technology                 |
| ----------------------- | -------------------------- |
| **Frontend**            | Streamlit                  |
| **LLM**                 | Groq (DeepSeek R1 Distill) |
| **Embeddings**          | HuggingFace (MiniLM)       |
| **Vector Store**        | FAISS                      |
| **Framework**           | LangChain                  |
| **Document Processing** | PyPDF + Unstructured       |
| **Logging**             | Python logging             |
| **Testing**             | Pytest                     |
| **Containerization**    | Docker                     |

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see [LICENSE.txt](LICENSE.txt) for details.

## 👤 Author

**Shishupal Rajpurohit**

* GitHub: [@ShishupalRajpurohit](https://github.com/ShishupalRajpurohit)
* LinkedIn: [Shishupal Rajpurohit](https://www.linkedin.com/in/shishupal-rajpurohit-039290190/)
* Portfolio: [Portfolio](https://sites.google.com/view/shishupals-portfolio/home)

## 🙏 Acknowledgments

- [LangChain](https://langchain.com) for the RAG framework
- [Groq](https://groq.com) for fast LLM inference
- [DeepSeek R1 Distill LLaMA 70B](https://www.groq.com/) for the model used in this project
- [HuggingFace MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) for embeddings
- [Streamlit](https://streamlit.io) for the web framework
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search

## 📞 Support

* 📧 Email: [shishupalrajpurohit2000@gmail.com](mailto:shishupalrajpurohit2000@gmail.com)
* 🐛 Issues: [GitHub Issues](https://github.com/ShishupalRajpurohit/BME_Bot/issues)
* 💬 Discussions: [GitHub Discussions](https://github.com/ShishupalRajpurohit/BME_Bot/discussions)

---

⭐ **Star this repo if you find it helpful!**

Built with ❤️ for the biomedical engineering community
