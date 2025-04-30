from abc import ABC, abstractmethod
from typing import Dict, Any, List
from ..throttling_manager import ThrottlingManager

class LLMApi(ABC):
    """Base class for LLM API interactions."""
    
    def __init__(self, config: Dict[str, Any], global_config: Dict[str, Any] = None):
        """
        Initialize with configuration.
        
        Args:
            config: LLM-specific configuration
            global_config: Global application configuration
        """
        self.api_key = config.get('api_key', '')
        self.model = config.get('model', '')
        self.api_url = config.get('api_url', '')
        self.num_completions = config.get('num_completions', 1)
        self.name = config.get('name', self.__class__.__name__)
        
        # Initialize throttling manager
        if global_config is None:
            global_config = {}
        self.throttling_manager = ThrottlingManager(global_config)
        
        # Apply throttling to internal _generate method
        self._generate_with_throttling = self.throttling_manager.with_throttling(
            self._generate,
            llm_name=self.name,
            llm_config=config
        )
    
    @abstractmethod
    def _generate(self, prompt: str) -> List[str]:
        """
        Internal method to generate responses from the LLM.
        This method should be implemented by subclasses and will be wrapped with throttling.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            A list of responses from the LLM
        """
        pass
    
    def generate(self, prompt: str) -> List[str]:
        """
        Generate one or more responses from the LLM with throttling applied.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            A list of responses from the LLM. The length of the list will be
            determined by the num_completions configuration, but may be less
            if the API cannot generate the requested number of completions.
        """
        return self._generate_with_throttling(prompt)
