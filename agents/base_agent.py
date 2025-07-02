from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from config import settings
import logging
import asyncio
from functools import partial

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
        self._loop = None
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
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the input and return results"""
        pass
    
    def process_sync(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous wrapper for process method"""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If called from async context, create a new task
            future = asyncio.ensure_future(self.process(input_data))
            return asyncio.run_coroutine_threadsafe(future, loop).result()
        else:
            # If called from sync context, run directly
            return asyncio.run(self.process(input_data))
    
    async def clear_memory(self):
        """Clear the agent's conversation memory"""
        self.memory.clear()
        logger.info(f"Cleared memory for {self.name} agent")
    
    def clear_memory_sync(self):
        """Synchronous wrapper for clear_memory"""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            future = asyncio.ensure_future(self.clear_memory())
            return asyncio.run_coroutine_threadsafe(future, loop).result()
        else:
            return asyncio.run(self.clear_memory())
    
    def _get_event_loop(self):
        """Get or create an event loop"""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
            return self._loop