import os  # To read environment variables like HuggingFace API token
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings, ChatHuggingFace  # For loading HuggingFace LLM and embeddings
from langchain_core.prompts import ChatPromptTemplate  # To create structured chat-style prompts
from langchain.chains.combine_documents import create_stuff_documents_chain  # Combines context documents with prompt
from langchain.chains import create_retrieval_chain  # Combines retriever with a QA chain
from langchain_community.vectorstores import FAISS  # For loading FAISS vectorstore locally
from dotenv import load_dotenv
from langchain_groq import ChatGroq  # Groq LLM wrapper

# Step 1: Setup LLM (Mistral via Hugging Face)
load_dotenv()
HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")  # Get HuggingFace API token from environment
hf_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"  # HF repo ID for the Mistral model

from langchain_groq import ChatGroq  # Groq integration

HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def load_llm(hf_repo_id):
    try:
        print("🔹 Switching to Groq backend...")
        chat_llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model="deepseek-r1-distill-llama-70b",  # You can also use "llama3-70b-8192" etc.
            temperature=0.5,
        )
        print("✅ Using Groq backend")
        
        return chat_llm
    except Exception as e:
        print(f"❌ Groq failed: {e}")
        print("🔹 Trying HuggingFace LLM...")
        llm = HuggingFaceEndpoint(
            repo_id=hf_repo_id,
            temperature=0.5,
            huggingfacehub_api_token=HF_TOKEN,
        )
        chat_llm = ChatHuggingFace(llm=llm)  # wrap as chat model
        print("✅ Using huggingface backend")
        return chat_llm




# Step 2: Define Prompt & Load FAISS Vector Store

# System instruction to keep answers concise, grounded in context only
custom_system_prompt ="""
You are a helpful assistant with two modes:

1. QA Mode (when the user asks a question about troubleshooting, errors, procedures, components, manuals, or technical topics):
   - ONLY use the provided CONTEXT to answer.
   - If the answer is not in the CONTEXT, reply exactly: "I don't know."
   - Do not use outside knowledge or make assumptions.
   - Do not add extra details or definitions beyond the CONTEXT.
   - Start the answer directly and be detailed.

2. Chat Mode (when the user is engaging in casual conversation, greetings, or non-technical small talk):
   - Respond naturally and conversationally.
   - You are not restricted to the CONTEXT here.

Context:
{context}
"""


# Define a structured multi-turn prompt with system and human roles
prompt = ChatPromptTemplate.from_messages([
    ("system", custom_system_prompt),  # Instruction message
    ("human", "{input}")  # User query will be passed as {input}
])

# Load FAISS vector DB
DB_FAISS_PATH = "vectorstore/db_faiss"  # Path to previously saved FAISS DB
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Same model used during indexing
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)  # Load FAISS with same embeddings

# Step 3: Create QA Chain with Retriever

llm = load_llm(hf_repo_id)  # Initialize LLM
retriever = db.as_retriever(search_kwargs={"k": 3})  # Set retriever to return top 3 similar chunks

# Combine LLM with prompt using "stuff" strategy (stuff all context into prompt)
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Connect retriever with the QA chain
chain = create_retrieval_chain(retriever, question_answer_chain)

# Step 4: Accept User Query and Run the Chain
user_query = input("Write Query Here: ")  # Take input from user
response = chain.invoke({"input": user_query})  # Run the full retrieval + QA pipeline

# Step 5: Show Output
print("\nResult:\n", response["answer"])  # Display the answer to the user
