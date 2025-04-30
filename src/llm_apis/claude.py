import anthropic
from typing import Dict, Any, List
from .base import LLMApi

class ClaudeApi(LLMApi):
    """API implementation for Claude."""
    
    def __init__(self, config: Dict[str, Any], global_config: Dict[str, Any] = None):
        """Initialize with Claude-specific configuration."""
        # Set the name before calling the parent constructor
        config['name'] = 'Claude'
        super().__init__(config, global_config)
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def _generate(self, prompt: str) -> List[str]:
        """
        Generate one or more responses from Claude.
        
        Args:
            prompt: The prompt to send to Claude
            
        Returns:
            A list of responses from Claude
        """
        responses = []
        
        # Unfortunately, Claude API doesn't natively support multiple completions in one call,
        # so we'll make multiple calls with different random seeds
        for i in range(self.num_completions):
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                # Use a different seed for each completion to ensure diversity
                # Note: Some APIs may not support this parameter
                system=f"Seed: {i}" if i > 0 else None
            )
            responses.append(message.content[0].text)
                
        # Ensure we return at least one response, even if empty
        if not responses:
            responses = ["Error: Failed to generate any responses from Claude"]
            
        return responses
