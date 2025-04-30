from typing import Dict, Any, List, Union

from .base_evaluator import BaseEvaluator
from .dimension_evaluator import DimensionEvaluator
from .bertscore_evaluator import BERTScoreEvaluator
from .bleu_evaluator import BLEUEvaluator
from .rouge_evaluator import ROUGEEvaluator
from .multi_evaluator import MultiEvaluator

def get_evaluator(config: Dict[str, Any]) -> BaseEvaluator:
    """
    Factory function to get the appropriate evaluator(s) based on configuration.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        An instance of the appropriate evaluator
    """
    # Check for the new array-based configuration
    evaluator_types = config.get('evaluator', {}).get('types', [])
    
    # If not specified, fall back to the legacy single type configuration
    if not evaluator_types:
        legacy_type = config.get('evaluator', {}).get('type', 'dimension')
        evaluator_types = [legacy_type]
    
    # If there's only one evaluator type, return it directly
    if len(evaluator_types) == 1:
        return _create_single_evaluator(evaluator_types[0], config)
    
    # If there are multiple evaluator types, create a MultiEvaluator
    evaluators = [_create_single_evaluator(evaluator_type, config) 
                 for evaluator_type in evaluator_types]
    
    return MultiEvaluator(config, evaluators)

def _create_single_evaluator(evaluator_type: str, config: Dict[str, Any]) -> BaseEvaluator:
    """
    Create a single evaluator of the specified type.
    
    Args:
        evaluator_type: Type of evaluator to create
        config: Full configuration dictionary
        
    Returns:
        An instance of the specified evaluator
    """
    if evaluator_type == 'dimension':
        return DimensionEvaluator(config)
    elif evaluator_type == 'bertscore':
        return BERTScoreEvaluator(config)
    elif evaluator_type == 'bleu':
        return BLEUEvaluator(config)
    elif evaluator_type == 'rouge':
        return ROUGEEvaluator(config)
    else:
        raise ValueError(f"Unsupported evaluator type: {evaluator_type}")
