import os
import yaml
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file and resolve environment variables.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    # Load raw config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Resolve environment variables in API keys
    for llm_name, llm_config in config.get('llms', {}).items():
        api_key = llm_config.get('api_key', '')
        if isinstance(api_key, str) and api_key.startswith('${') and api_key.endswith('}'):
            env_var = api_key[2:-1]
            llm_config['api_key'] = os.environ.get(env_var, '')
    
    return config
