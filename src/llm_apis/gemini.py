import google.generativeai as genai
from typing import Dict, Any, List
from .base import LLMApi
import re

class GeminiApi(LLMApi):
    """API implementation for Gemini."""
    
    def __init__(self, config: Dict[str, Any], global_config: Dict[str, Any] = None):
        """Initialize with Gemini-specific configuration."""
        # Set the name before calling the parent constructor
        config['name'] = 'Gemini'
        super().__init__(config, global_config)
        
        # Configure the API with the key
        genai.configure(api_key=self.api_key)
        
        # Store the configured model name
        self.configured_model_name = self.model
        self.model_instance = None
        
        try:
            # First, try to use the explicitly configured model
            try:
                print(f"Attempting to initialize with configured model: {self.configured_model_name}")
                self.model_instance = genai.GenerativeModel(model_name=self.configured_model_name)
                print(f"Successfully initialized model: {self.configured_model_name}")
                return
            except Exception as model_error:
                print(f"Could not initialize configured model: {model_error}")
                
            # If that fails, list available models and select one
            print("Listing available Gemini models:")
            available_models = list(genai.list_models())
            
            # Print available models for debugging
            for model in available_models:
                model_name = model.name
                # Extract just the model name without the full path
                short_name = model_name.split('/')[-1] if '/' in model_name else model_name
                
                if hasattr(model, "supported_generation_methods"):
                    methods = model.supported_generation_methods
                else:
                    methods = "Unknown"
                print(f"  - {short_name}: {methods}")
            
            # Define a priority order for model selection
            priority_models = [
                "gemini-1.5-flash",
                "gemini-1.5-pro",
                "gemini-1.0-pro"
            ]
            
            # Try to find models in priority order
            selected_model = None
            for priority_model in priority_models:
                for model in available_models:
                    # Extract the short model name without the full path
                    model_name = model.name
                    short_name = model_name.split('/')[-1] if '/' in model_name else model_name
                    
                    # Check if this is a matching model
                    if priority_model in short_name:
                        selected_model = model
                        print(f"Found priority model: {short_name}")
                        break
                
                if selected_model:
                    break
            
            # If no priority models found, try any model that supports text generation
            if not selected_model:
                for model in available_models:
                    if hasattr(model, "supported_generation_methods") and "generateContent" in model.supported_generation_methods:
                        selected_model = model
                        print(f"Using model with generateContent support: {model.name}")
                        break
            
            # If we found a suitable model, initialize it
            if selected_model:
                print(f"Initializing with model: {selected_model.name}")
                self.model_instance = genai.GenerativeModel(model_name=selected_model.name)
            else:
                print("No suitable Gemini models found")
                self.model_instance = None
                
        except Exception as e:
            print(f"Error initializing Gemini API: {e}")
            self.model_instance = None
    
    def _generate(self, prompt: str) -> List[str]:
        """
        Generate one or more responses from Gemini.
        
        Args:
            prompt: The prompt to send to Gemini
            
        Returns:
            A list of responses from Gemini
        """
        if self.model_instance is None:
            return ["Error: Could not initialize Gemini model. Check API key and available models."]
            
        responses = []
        
        # Configure generation parameters
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
            # Note: candidate_count is supported in some Gemini models
            "candidate_count": min(self.num_completions, 8)  # Limited to 8
        }
        
        # Try to generate multiple responses in one call if supported
        response = self.model_instance.generate_content(
            contents=prompt,
            generation_config=generation_config
        )
        
        # Check if we got multiple candidates
        if hasattr(response, 'candidates') and len(response.candidates) > 1:
            # Extract text from each candidate
            for candidate in response.candidates:
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    text = ''.join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
                    responses.append(text)
        else:
            # Extract the text from a single response
            if hasattr(response, 'text'):
                responses.append(response.text)
            elif hasattr(response, 'parts'):
                text = ''.join(part.text for part in response.parts if hasattr(part, 'text'))
                responses.append(text)
            else:
                responses.append(str(response))
            
            # If multiple completions were requested but only one was returned,
            # make additional calls with different temperatures
            for i in range(1, self.num_completions):
                # Vary the temperature for diversity
                gen_config_variation = generation_config.copy()
                gen_config_variation["temperature"] = min(0.9, generation_config["temperature"] + i * 0.1)
                
                resp = self.model_instance.generate_content(
                    contents=prompt,
                    generation_config=gen_config_variation
                )
                
                if hasattr(resp, 'text'):
                    responses.append(resp.text)
                elif hasattr(resp, 'parts'):
                    text = ''.join(part.text for part in resp.parts if hasattr(part, 'text'))
                    responses.append(text)
                else:
                    responses.append(str(resp))
        
        # Ensure we have at least one response
        if not responses:
            responses.append("Failed to generate any responses from Gemini")
            
        return responses
