"""
QA Chain management module
Handles question-answering chain creation and execution
"""
from typing import Dict, Any, Optional
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from utils.logger import setup_logger

logger = setup_logger(__name__)

class QAChainManager:
    """Manages question-answering chain operations"""
    
    def __init__(
        self,
        groq_api_key: str,
        model_name: str = "deepseek-r1-distill-llama-70b",
        temperature: float = 0.0
    ):
        """
        Initialize QA chain manager
        
        Args:
            groq_api_key: Groq API key
            model_name: LLM model name
            temperature: LLM temperature (0.0 = deterministic)
        """
        self.groq_api_key = groq_api_key
        self.model_name = model_name
        self.temperature = temperature
        self._llm = None
        self._qa_chain = None
    
    @property
    def llm(self) -> ChatGroq:
        """Lazy load LLM"""
        if self._llm is None:
            logger.info(f"Initializing LLM: {self.model_name}")
            try:
                self._llm = ChatGroq(
                    model_name=self.model_name,
                    temperature=self.temperature,
                    groq_api_key=self.groq_api_key,
                )
                logger.info("LLM initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize LLM: {str(e)}")
                raise
        return self._llm
    
    def create_qa_chain(
        self,
        retriever,
        prompt_template: str,
        return_source_documents: bool = True
    ) -> ConversationalRetrievalChain:
        """
        Create ConversationalRetrievalChain
        
        Args:
            retriever: LangChain retriever instance
            prompt_template: Custom prompt template string
            return_source_documents: Whether to return source documents
            
        Returns:
            Configured ConversationalRetrievalChain
        """
        try:
            logger.info("Creating QA chain")
            
            # Create prompt
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Add memory
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            
            # Create chain
            self._qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=memory,
                return_source_documents=return_source_documents,
                combine_docs_chain_kwargs={"prompt": prompt}
            )
            
            logger.info("QA chain created successfully")
            return self._qa_chain
            
        except Exception as e:
            logger.error(f"Failed to create QA chain: {str(e)}")
            raise
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Execute query through QA chain
        
        Args:
            question: User question
            
        Returns:
            Dictionary containing 'answer' and optionally 'source_documents'
        """
        if self._qa_chain is None:
            raise ValueError("QA chain not initialized. Call create_qa_chain first.")
        
        try:
            logger.info(f"Processing query: {question[:50]}...")
            response = self._qa_chain.invoke({'question': question})
            logger.info("Query processed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise
    
    def format_response(
        self,
        response: Dict[str, Any],
        include_sources: bool = False
    ) -> str:
        """
        Format response for display
        
        Args:
            response: Raw response from QA chain
            include_sources: Whether to include source information
            
        Returns:
            Formatted response string
        """
        result = response.get("answer", "No answer generated.")
        
        # Remove <think> tags and their content
        import re
        result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)
        result = result.strip()
        
        if not include_sources:
            return result
        
        source_docs = response.get("source_documents", [])
        if not source_docs:
            return result
        
        sources_text = "\n\n---\n**Sources:**\n"
        for i, doc in enumerate(source_docs[:3], 1):  # Show top 3 sources
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            sources_text += f"{i}. {source} (Page {page})\n"
        
        return result + sources_text