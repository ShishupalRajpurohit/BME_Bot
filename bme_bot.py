# 📦 Import standard and external libraries
import os  # For accessing environment variables
import streamlit as st  # For building interactive web UI

# 🧠 Import LangChain modules
from langchain_huggingface import HuggingFaceEmbeddings  # Embedding model from HuggingFace
from langchain.chains import RetrievalQA  # Chain for retrieval-based question answering

from langchain_community.vectorstores import FAISS  # Vector store (FAISS)
from langchain_core.prompts import PromptTemplate  # For customizing prompts
from langchain_huggingface import HuggingFaceEndpoint  # Optional: LLM via HuggingFace Hub
from langchain_groq import ChatGroq  # Groq LLM wrapper

# 🔐 Load environment variables from .env file
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  # Automatically finds and loads the .env file

# 🔍 Path to the FAISS vector store directory
DB_FAISS_PATH = "vectorstore/db_faiss"

# 📥 Load vector store using cached Streamlit resource
@st.cache_resource
def get_vectorstore():
    # Use MiniLM model to create embedding function
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    # Load existing FAISS vector store from disk
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# 🧾 Create a custom prompt template for retrieval-based QA
def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]  # These variables will be replaced during runtime
    )
    return prompt

# 🚫 Currently not used, but useful if you switch to HuggingFace LLM
def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )
    return llm

# 🎯 Main function for running Streamlit chatbot
def main():
    st.title("Ask Chatbot!")  # Display the title in the web UI

    # 🗨️ Initialize message history if not present
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # 📜 Display previous chat history
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # ⌨️ User input field
    prompt = st.chat_input("Pass your prompt here")

    if prompt:  # If user submits a prompt
        st.chat_message('user').markdown(prompt)  # Display it in the chat
        st.session_state.messages.append({'role': 'user', 'content': prompt})  # Save it in session state

        # 🧾 Custom prompt template
        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer user's question.
        If you dont know the answer, just say that you dont know, dont try to make up an answer. 
        Dont provide anything out of the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """

        try:
            # 📥 Load vector store
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            # 🔗 Create RetrievalQA chain using Groq-hosted LLaMA-4 model
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGroq(
                    model_name="meta-llama/llama-4-maverick-17b-128e-instruct",  # Fast, free model from Groq
                    temperature=0.0,
                    groq_api_key=os.environ["GROQ_API_KEY"],  # API key from your .env file
                ),
                chain_type="stuff",  # Simple RAG logic (concatenate chunks)
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),  # Retrieve top 3 relevant chunks
                return_source_documents=True,  # Return chunks used in the response
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}  # Use custom prompt
            )

            # 🧠 Invoke the full pipeline with user input
            response = qa_chain.invoke({'query': prompt})

            # 📤 Prepare and display the result
            result = response["result"]
            source_documents = response["source_documents"]  # List of source docs retrieved
            result_to_show = result + "\nSource Docs:\n" + str(source_documents)

            st.chat_message('assistant').markdown(result_to_show)  # Show the answer
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})  # Save to history

        except Exception as e:
            # ❌ Handle and show any error
            st.error(f"Error: {str(e)}")

# 🚀 Run the Streamlit app
if __name__ == "__main__":
    main()
