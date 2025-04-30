from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseEvaluator(ABC):
    """Base class for all evaluators."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with configuration.
        
        Args:
            config: Full configuration dictionary
        """
        self.config = config
    
    @abstractmethod
    def evaluate(self, reference: str, input_text: str) -> Dict[str, Any]:
        """
        Evaluate an input text against a reference text.
        
        Args:
            reference: Reference text
            input_text: Input text to evaluate
            
        Returns:
            Dictionary of evaluation results
        """
        pass
