from typing import Dict, Any, List
from .base_evaluator import BaseEvaluator

class MultiEvaluator(BaseEvaluator):
    """Runs multiple evaluators and aggregates their results."""
    
    def __init__(self, config: Dict[str, Any], evaluators: List[BaseEvaluator]):
        """
        Initialize with configuration and a list of evaluators.
        
        Args:
            config: Full configuration dictionary
            evaluators: List of evaluator instances to run
        """
        super().__init__(config)
        self.evaluators = evaluators
        self.evaluator_types = [evaluator.__class__.__name__ for evaluator in self.evaluators]
    
    def evaluate(self, reference: str, input_text: str) -> Dict[str, Any]:
        """
        Evaluate an input text using all configured evaluators.
        
        Args:
            reference: Reference text
            input_text: Input text to evaluate
            
        Returns:
            Dictionary of results from all evaluators
        """
        results = {
            'evaluator_type': 'multi',
            'evaluators_used': self.evaluator_types,
            'results': {}
        }
        
        # Run each evaluator and collect results
        for evaluator in self.evaluators:
            evaluator_name = evaluator.__class__.__name__
            # Extract the type name from class (e.g., DimensionEvaluator -> dimension)
            evaluator_type = evaluator_name.replace('Evaluator', '').lower()
            
            try:
                # Run this evaluator
                evaluator_results = evaluator.evaluate(reference, input_text)
                # Store results under the evaluator type key
                results['results'][evaluator_type] = evaluator_results
                print(f"Completed evaluation with {evaluator_name}")
            except Exception as e:
                print(f"Error running {evaluator_name}: {e}")
                # Add error information to results
                results['results'][evaluator_type] = {
                    'error': str(e),
                    'evaluator_type': evaluator_type
                }
        
        return results
