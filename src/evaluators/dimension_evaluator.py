import os
from typing import Dict, Any, List
from ..llm_apis import get_llm_api
from .base_evaluator import BaseEvaluator

class DimensionEvaluator(BaseEvaluator):
    """Evaluates input texts using LLMs across multiple dimensions."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with configuration.
        
        Args:
            config: Full configuration dictionary
        """
        super().__init__(config)
        self.prompt_cache = {}  # Cache for loaded prompts
        
        # Get the active LLM name from config
        self.active_llm = config.get('active_llm', 'claude')  # Default to claude if not specified
        
        # Check if the active LLM is configured
        if self.active_llm not in config.get('llms', {}):
            raise ValueError(f"Active LLM '{self.active_llm}' is not configured in the llms section")
        
        # Initialize only the active LLM API
        llm_config = config.get('llms', {}).get(self.active_llm, {})
        self.llm_api = get_llm_api(self.active_llm, llm_config, config)

        # Obtain the model version
        if self.active_llm == 'gemini':
            get_name_func = getattr(self.llm_api, 'get_model_name', None)
            if callable(get_name_func):
                detailed_model_name = get_name_func()
                if detailed_model_name.startswith("gemini-"):
                    detailed_model_name = detailed_model_name[len("gemini-"):]
                self.active_llm = f"gemini-{detailed_model_name}"
        
        # Get the number of completions to generate
        self.num_completions = llm_config.get('num_completions', 1)
    
    def evaluate(self, reference: str, input_text: str) -> Dict[str, Any]:
        """
        Evaluate an input text across all dimensions using the active LLM.
        
        Args:
            reference: Reference text
            input_text: Input text to evaluate
            
        Returns:
            Dictionary of results by dimension
        """
        llm_name = self.active_llm
        if hasattr(self.llm_api, 'active_model_name'):
            llm_name = self.llm_api.active_model_name

        possible_categories = ["Spoofing", "Tampering", "Repudiation", "Information Disclosure", "Elevation of Privilege", "Denial of Service"]

        text_combined = f"{reference} {input_text}".lower()
        inferred_category = None
        first_position = float('inf')

        for category in possible_categories:
            pos = text_combined.find(category.lower())
            if pos != -1 and pos < first_position:
                inferred_category = category
                first_position = pos

        results = {
            'evaluator_type': 'dimension',
            'llm': llm_name,
            'num_completions': self.num_completions,
            'dimensions': {}
        }
        
        for dim_name, dim_config in self.config.get('dimensions', {}).items():
            dim_category = dim_config.get('category', 'All')

            if dim_category != 'All' and dim_category != inferred_category:
                continue

            prompt_file = dim_config.get('prompt_file')
            if not prompt_file:
                continue

            weight = dim_config.get('weight', 0.0)
            prompt_template = self._load_prompt_from_file(prompt_file)
            prompt = prompt_template.format(reference=reference, input=input_text)
            responses = self.llm_api.generate(prompt)

            results['dimensions'][dim_name] = {
                'responses': responses,
                'weight': weight
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
