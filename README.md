# 🧠💬 BME_Bot: AI-Powered Biomedical Troubleshooting Assistant

**BME_Bot** is an intelligent, interactive chatbot tailored for **biomedical engineers** and **healthcare professionals**, with a specialized focus on **dialysis machine troubleshooting** and answering **basic medical queries**.

Using a powerful combination of **LangChain**, **Streamlit**, **FAISS**, and open-source **LLMs**, BME_Bot brings instant, reliable, and context-aware support straight from service manuals and verified medical knowledge.

---

## 🔧 What It Can Do

- 🩺 **Troubleshoot Dialysis Machines**  
  Resolve hardware or alarm issues using real-time insights extracted from embedded **service manuals**.

- 📚 **Answer Basic Medical Questions**  
  Get concise, contextual answers to general biomedical queries.

- 💬 **User-friendly Chat Interface**  
  Interact via an intuitive **Streamlit UI** — no coding knowledge needed!

---

## 🧠 How It Works (Under the Hood)

### 🔍 Vector Search with FAISS
- All medical and service documents are converted into embeddings using:
  ```python
  HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
  ```
- Stored and retrieved using **FAISS**, a high-performance vector store.

### 🤖 LLMs Used

| Model                             | Provider | Notes                         |
|----------------------------------|----------|-------------------------------|
| `meta-llama/llama-4-maverick-17b-128e-instruct` | [Groq](https://console.groq.com) | 🚀 Free, ultra-fast via `ChatGroq` |
| `mistralai/Mistral-7B-Instruct-v0.3`            | HuggingFace | 🔒 Paid API access required     |

### 🧠 Retrieval-Augmented Generation (RAG)
- Powered by `RetrievalQA.from_chain_type()` from **LangChain**
- Uses a **custom prompt template** to ensure precise, context-bound responses
- Returns both the **answer** and **source documents**

---

## 🛠️ Tech Stack

| Component        | Purpose                                      |
|------------------|----------------------------------------------|
| **LangChain**     | Chaining LLMs with data and retrieval tools |
| **Streamlit**     | Lightweight UI for web-based interaction     |
| **FAISS**         | Semantic search over embedded documents      |
| **HuggingFace**   | For embedding models (MiniLM)                |
| **Groq**          | Ultra-fast LLM inference backend             |
| **.env**          | Secure API key management                    |

---

## 🚀 How to Run Locally

```bash
# 1. Clone this repository
git clone https://github.com/ShishupalRajpurohit/BME_Bot.git
cd BME_Bot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API keys in a `.env` file
# Example:
# GROQ_API_KEY=your-groq-api-key
# HF_TOKEN=your-huggingface-token

# 4. Start the app
streamlit run bme_bot.py
```

---

## 💡 Example Interactions

```text
👨‍🔧 Engineer: How to fix conductivity error in dialysis machine?
🤖 BME_Bot: Start by verifying the calibration via the 'Conductivity Check' section...
             [Source: Fresenius 4008 Hemodialysis System - Technical manual.pdf]

🩺 User: What is hemodialysis?
🤖 BME_Bot: Hemodialysis is a process that filters blood using a dialyzer...
             [Source: The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf]
```

---

## 📌 Highlights

- ✅ Real-world **dialysis service manuals** embedded
- ✅ Smart responses with **LLMs + context**
- ✅ Runs **locally** with Streamlit
- ✅ **Extendable** to other medical devices or hospital systems


---

## 🤝 Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)
- [Groq](https://groq.com/)
- [HuggingFace](https://huggingface.co/)
- [Streamlit](https://streamlit.io/)

---

> ⚕️ Built with ❤️ by a Former Biomedical Engineer for the Biomedical community.
