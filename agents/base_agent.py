from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from config import settings
import logging

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Base class for all agents in the system"""
    
    def __init__(self, name: str, model_name: Optional[str] = None):
        self.name = name
        self.model_name = model_name or settings.ollama_model
        self.llm = self._initialize_llm()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.executor = None
        logger.info(f"Initialized {self.name} agent with model {self.model_name}")
    
    def _initialize_llm(self) -> ChatOllama:
        """Initialize the Ollama LLM"""
        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=self.model_name,
            temperature=0.1,
            num_predict=4096
        )
    
    @abstractmethod
    def create_prompt(self) -> Any:
        """Create the prompt template for the agent"""
        pass
    
    @abstractmethod
    def create_tools(self) -> list:
        """Create the tools for the agent"""
        pass
    
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the input and return results"""
        pass
    
    def clear_memory(self):
        """Clear the agent's conversation memory"""
        self.memory.clear()
        logger.info(f"Cleared memory for {self.name} agent")