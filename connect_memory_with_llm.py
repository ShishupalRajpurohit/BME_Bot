''' Connect Memory With LLM (Updated for LangChain 0.1.17+) '''

import os  # To read environment variables like HuggingFace API token
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings  # For loading HuggingFace LLM and embeddings
from langchain_core.prompts import ChatPromptTemplate  # To create structured chat-style prompts
from langchain.chains.combine_documents import create_stuff_documents_chain  # Combines context documents with prompt
from langchain.chains import create_retrieval_chain  # Combines retriever with a QA chain
from langchain_community.vectorstores import FAISS  # For loading FAISS vectorstore locally

# Step 1: Setup LLM (Mistral via Hugging Face)
HF_TOKEN = os.environ.get("HF_TOKEN")  # Get HuggingFace API token from environment
hf_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"  # HF repo ID for the Mistral model

def load_llm(hf_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=hf_repo_id,                       # HF model
        temperature=0.5,                          # Controls randomness
        huggingfacehub_api_token=HF_TOKEN         # Auth token
        )
    return llm



# Step 2: Define Prompt & Load FAISS Vector Store

# System instruction to keep answers concise, grounded in context only
custom_system_prompt = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Don't provide anything outside the given context.

Context: {context}
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
