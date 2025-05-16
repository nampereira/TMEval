import os
from typing import Dict, Any, List
from .base_evaluator import BaseEvaluator
import torch
import warnings
import logging

class BERTScoreEvaluator(BaseEvaluator):
    """Evaluates input texts using BERTScore metrics with a local model."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with configuration.
        
        Args:
            config: Full configuration dictionary
        """
        super().__init__(config)
        self.model_name = config.get('evaluator', {}).get('bertscore', {}).get('model', 'roberta-large')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.idf = config.get('evaluator', {}).get('bertscore', {}).get('idf', True)
        self.rescale_with_baseline = config.get('evaluator', {}).get('bertscore', {}).get('rescale_with_baseline', True)
        self.verbose = config.get('evaluator', {}).get('bertscore', {}).get('verbose', False)
        
        # Filter out transformer model initialization warnings if not in verbose mode
        if not self.verbose:
            # Filter transformers model warnings
            logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
            # Suppress specific warnings about weights not initialized
            warnings.filterwarnings(
                "ignore", 
                message="Some weights of .* were not initialized from the model checkpoint .*"
            )
        
        # Import bert_score only when needed to avoid loading the model until evaluation
        self.bert_score = None
        
        # Log initialization information
        print(f"Initializing BERTScore evaluator with model: {self.model_name}")
        print(f"Using device: {self.device}")
        print(f"Note: Initialization warnings about model weights are expected and can be safely ignored.")
    
    def evaluate(self, reference: str, input_text: str) -> Dict[str, Any]:
        """
        Evaluate an input text using BERTScore metrics.
        
        Args:
            reference: Reference text
            input_text: Input text to evaluate
            
        Returns:
            Dictionary of BERTScore results
        """
        # Import bert_score only when needed
        if self.bert_score is None:
            try:
                import bert_score
                self.bert_score = bert_score
                print(f"Successfully imported bert_score library")
            except ImportError:
                raise ImportError("bert-score package is not installed. Please install it with 'pip install bert-score'")
        
        # Clean the reference and input texts
        reference = self._clean_text(reference)
        input = self._clean_text(input_text)

        # Split texts into sentences for more granular evaluation
        reference_sentences = self._split_into_sentences(reference)
        input_sentences = self._split_into_sentences(input)
        
        print(f"Processing {len(reference_sentences)} reference sentences and {len(input_sentences)} input sentences")
        
        # # Handle potential mismatch in sentence counts
        # if len(reference_sentences) != len(input_sentences):
        #     print(f"Warning: Number of sentences doesn't match. Reference has {len(reference_sentences)}, Input has {len(input_sentences)}")
        #     # For BERTScore, we can still calculate the scores even with different numbers of sentences
        
        # Calculate BERTScore
        try:
            print(f"Calculating BERTScore using {self.model_name}...")
            
            # Temporarily disable specific warnings during calculation
            with warnings.catch_warnings():
                if not self.verbose:
                    warnings.filterwarnings(
                        "ignore", 
                        message="Some weights of .* were not initialized from the model checkpoint .*"
                    )
                
                # Prepare sentence-level results
                sentence_results = []
                f1_scores_list = []

                for input_sentence in input_sentences:
                    # Repeat the input sentence to match the number of reference sentences
                    cands = [input_sentence] * len(reference_sentences)
                    refs = reference_sentences

                    P, R, F1 = self.bert_score.score(
                        cands=cands,
                        refs=refs,
                        model_type=self.model_name,
                        device=self.device,
                        lang="en",  # Default to English, could be made configurable
                        verbose=self.verbose,  # Controls showing progress
                        idf=False,
                        rescale_with_baseline=self.rescale_with_baseline  # Scales scores to be more interpretable
                    )

                    # Convert PyTorch tensors to Python lists
                    precision_scores = P.tolist()
                    recall_scores = R.tolist()
                    f1_scores = F1.tolist()

                    # Find the best matching reference sentence
                    max_f1 = max(f1_scores)
                    max_index = f1_scores.index(max_f1)

                    sentence_results.append({
                        'input': input_sentence,
                        'best_reference': reference_sentences[max_index],
                        'precision': float(precision_scores[max_index]),
                        'recall': float(recall_scores[max_index]),
                        'f1': float(f1_scores[max_index])
                    })

                    f1_scores_list.append(f1_scores[max_index])

                # Calculate overall scores (average)
                overall_f1 = float(sum(f1_scores_list) / len(f1_scores_list))

                print(f"BERTScore calculation complete")
                
                # Prepare and return results
                results = {
                    'evaluator_type': 'bertscore',
                    'implementation': 'local-model',
                    'model': self.model_name,
                    'idf': self.idf,
                    'rescale_with_baseline': self.rescale_with_baseline,
                    'scores': {
                        'overall': {
                            'f1': overall_f1
                        },
                        'sentence_level': sentence_results
                    }
                }
                
                return results
                
        except Exception as e:
            print(f"Error calculating BERTScore: {e}")
            # Return error information
            return {
                'evaluator_type': 'bertscore',
                'implementation': 'local-model',
                'model': self.model_name,
                'error': str(e)
            }
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences for more granular evaluation.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Try to use NLTK if available for better sentence splitting
        # try:
        #     from nltk.tokenize import sent_tokenize
        #     try:
        #         return sent_tokenize(text)
        #     except LookupError:
        #         # NLTK resources might not be downloaded
        #         import nltk
        #         nltk.download('punkt')
        #         return sent_tokenize(text)
        # except ImportError:
        #     # Fall back to simple rule-based splitting if NLTK is not available
        #     import re
        #     sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        #     return [s.strip() for s in sentences if s.strip()]

        return [line.strip() for line in text.strip().split('\n') if line.strip()]
