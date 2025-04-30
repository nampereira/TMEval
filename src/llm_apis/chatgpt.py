import openai
from typing import Dict, Any, List
from .base import LLMApi

class ChatGPTApi(LLMApi):
    """API implementation for ChatGPT."""
    
    def __init__(self, config: Dict[str, Any], global_config: Dict[str, Any] = None):
        """Initialize with ChatGPT-specific configuration."""
        # Set the name before calling the parent constructor
        config['name'] = 'ChatGPT'
        super().__init__(config, global_config)
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def _generate(self, prompt: str) -> List[str]:
        """
        Generate one or more responses from ChatGPT.
        
        Args:
            prompt: The prompt to send to ChatGPT
            
        Returns:
            A list of responses from ChatGPT
        """
        # OpenAI API supports n parameter for multiple completions in a single call
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            n=self.num_completions,
            temperature=0.7  # Add some randomness for diversity in completions
        )
        
        # Extract all choices from the response
        responses = [choice.message.content for choice in response.choices]
        return responses
