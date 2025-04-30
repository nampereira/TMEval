import os
from typing import Dict, Any, List
from .llm_apis import get_llm_api

class Evaluator:
    """Handles evaluation of input texts using the configured LLM."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with configuration.
        
        Args:
            config: Full configuration dictionary
        """
        self.config = config
        self.prompt_cache = {}  # Cache for loaded prompts
        
        # Get the active LLM name from config
        self.active_llm = config.get('active_llm', 'claude')  # Default to claude if not specified
        
        # Check if the active LLM is configured
        if self.active_llm not in config.get('llms', {}):
            raise ValueError(f"Active LLM '{self.active_llm}' is not configured in the llms section")
        
        # Initialize only the active LLM API
        llm_config = config.get('llms', {}).get(self.active_llm, {})
        self.llm_api = get_llm_api(self.active_llm, llm_config)
    
    def evaluate(self, reference: str, input_text: str) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate an input text across all dimensions using the active LLM.
        
        Args:
            reference: Reference text
            input_text: Input text to evaluate
            
        Returns:
            Dictionary of results by dimension
        """
        results = {}
        
        for dim_name, dim_config in self.config.get('dimensions', {}).items():
            prompt_file = dim_config.get('prompt_file', '')
            prompt_template = self._load_prompt_from_file(prompt_file)
            
            prompt = prompt_template.format(reference=reference, input=input_text)
            
            response = self.llm_api.generate(prompt)
            
            results[dim_name] = {
                'response': response,
                'weight': dim_config.get('weight', 0.25)
            }
        
        return results
    
    def _load_prompt_from_file(self, prompt_file: str) -> str:
        """
        Load a prompt template from a file, with caching.
        
        Args:
            prompt_file: Path to the prompt file
            
        Returns:
            The prompt template as a string
        """
        # Check if prompt is already in cache
        if prompt_file in self.prompt_cache:
            return self.prompt_cache[prompt_file]
        
        # Load prompt from file
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt_template = f.read()
                self.prompt_cache[prompt_file] = prompt_template
                return prompt_template
        except Exception as e:
            print(f"Error loading prompt file {prompt_file}: {e}")
            # Return a basic fallback prompt if file loading fails
            fallback = "Please evaluate this text. Reference: {reference} Input: {input}"
            return fallback
