import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

class ResultsManager:
    """Manages result aggregation and saving."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with configuration.
        
        Args:
            config: Full configuration dictionary
        """
        self.config = config
        self.output_format = config.get('output', {}).get('format', 'json')
        self.output_directory = config.get('output', {}).get('directory', 'results')
        self.filename_template = config.get('output', {}).get('filename_template', '{input_filename}_eval.json')
        
        # Get evaluator type(s)
        evaluator_types = config.get('evaluator', {}).get('types', [])
        if not evaluator_types:
            evaluator_types = [config.get('evaluator', {}).get('type', 'dimension')]
        
        self.evaluator_types = evaluator_types
        self.is_multi_evaluator = len(evaluator_types) > 1
    
    def save_results(self, 
                    results: Dict[str, Any], 
                    reference: str, 
                    input_text: str, 
                    reference_filename: Optional[str] = None,
                    input_filename: Optional[str] = None,
                    title: str = "",
                    description: str = "",
                    tags: Optional[List[str]] = None) -> str:
        """
        Save evaluation results to a file.
        
        Args:
            results: Evaluation results 
            reference: Reference text
            input_text: Input text
            reference_filename: Name of the reference file (optional)
            input_filename: Name of the input file (optional)
            title: Title of the text pair (optional)
            description: Description of the text pair (optional)
            tags: List of tags for the text pair (optional)
            
        Returns:
            Path to the saved results file
        """
        # Initialize tags if None
        if tags is None:
            tags = []
            
        # Create result data structure
        result_data = {
            'timestamp': datetime.now().isoformat(),
            'reference': {
                'text': reference,
                'filename': reference_filename
            },
            'input': {
                'text': input_text,
                'filename': input_filename
            },
            'metadata': {
                'title': title,
                'description': description,
                'tags': tags
            },
            'results': results
        }
        
        # Compute average value
        dimensions = results.get('dimensions', {})
        w_avg = 0
        for dim in dimensions:
            sum = 0
            values = dimensions[dim].get('responses', {})
            for val in values: 
                sp = val.split(":",1) # deal with responses in the format "Response: <int>"
                val = sp[0] if len(sp) == 1 else sp[1]
                sum = sum + int(val.rstrip())
            avg = sum / len(values) 
            dimensions[dim]['average'] = avg
            w_avg = w_avg + avg * dimensions[dim].get('weight', 0)
        dimensions['score'] = w_avg
                
        # Create output directory if it doesn't exist
        os.makedirs(self.output_directory, exist_ok=True)
        
        # Determine output filename
        if input_filename:
            base_filename = os.path.splitext(os.path.basename(input_filename))[0]
            
            # Add evaluator type to filename
            if self.is_multi_evaluator:
                evaluator_suffix = "_multi"
            elif 'evaluator_type' in results:
                evaluator_suffix = f"_{results['evaluator_type']}"
            else:
                evaluator_suffix = f"_{self.evaluator_types[0]}"
                
            output_filename = self.filename_template.format(
                input_filename=f"{base_filename}{evaluator_suffix}"
            )
        else:
            # Generate a timestamp-based filename for direct input
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if self.is_multi_evaluator:
                output_filename = f"eval_multi_{timestamp}.json"
            elif 'evaluator_type' in results:
                output_filename = f"eval_{results['evaluator_type']}_{timestamp}.json"
            else:
                output_filename = f"eval_{self.evaluator_types[0]}_{timestamp}.json"
        
        output_path = os.path.join(self.output_directory, output_filename)
        
        # Save results
        if self.output_format == 'json':
            with open(output_path, 'w') as f:
                json.dump(result_data, f, indent=2)
        else:
            # Implement other formats if needed
            raise ValueError(f"Unsupported output format: {self.output_format}")
        
        return output_path
