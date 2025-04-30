from typing import Dict, Any
from .base import LLMApi
from .claude import ClaudeApi
from .chatgpt import ChatGPTApi
from .gemini import GeminiApi

def get_llm_api(llm_name: str, config: Dict[str, Any], global_config: Dict[str, Any] = None) -> LLMApi:
    """
    Factory function to get the appropriate LLM API.
    
    Args:
        llm_name: Name of the LLM (claude, chatgpt, gemini)
        config: LLM-specific configuration
        global_config: Global application configuration
        
    Returns:
        An instance of the appropriate LLM API
    """
    llm_map = {
        'claude': ClaudeApi,
        'chatgpt': ChatGPTApi,
        'gemini': GeminiApi
    }
    
    if llm_name not in llm_map:
        raise ValueError(f"Unsupported LLM: {llm_name}")
    
    return llm_map[llm_name](config, global_config)
