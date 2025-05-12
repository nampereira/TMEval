import os
from typing import Dict, Any, List
from .base_evaluator import BaseEvaluator
import nltk
import sys

class ROUGEEvaluator(BaseEvaluator):
    """Evaluates input texts using ROUGE score metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with configuration.
        
        Args:
            config: Full configuration dictionary
        """
        super().__init__(config)
        self.rouge_config = config.get('evaluator', {}).get('rouge', {})
        self.use_stemmer = self.rouge_config.get('use_stemmer', True)
        self.rouge_types = self.rouge_config.get('rouge_types', ['rouge1', 'rouge2', 'rougeL'])
        
        # Check if rouge_score is installed
        try:
            import rouge_score
            self.rouge_score_available = True
        except ImportError:
            self.rouge_score_available = False
            print("\n==============================================================================")
            print("ROUGE evaluator requires the rouge_score package, which is not installed.")
            print("Please install it using:")
            print("  pip install rouge-score")
            print("==============================================================================\n")
        
        # Check if NLTK resources are available
        self.nltk_resources_available = self._check_nltk_resources()
        
        if not self.nltk_resources_available:
            print("\n==============================================================================")
            print("ROUGE evaluator requires NLTK data resources that are not installed.")
            print("Please download them using:")
            print("  python download_nltk_data.py")
            print("  or")
            print("  import nltk; nltk.download('punkt')")
            print("==============================================================================\n")
        
        # Print initialization info
        if self.rouge_score_available and self.nltk_resources_available:
            print(f"Initializing ROUGE evaluator")
            print(f"Rouge types: {self.rouge_types}")
            print(f"Use stemmer: {self.use_stemmer}")
    
    def _check_nltk_resources(self):
        """Check if required NLTK resources are available."""
        try:
            nltk.data.find('tokenizers/punkt')
            return True
        except LookupError:
            try:
                # Try to download punkt automatically
                print("Attempting to download NLTK punkt tokenizer...")
                nltk.download('punkt', quiet=True)
                # Check again
                nltk.data.find('tokenizers/punkt')
                print("Successfully downloaded NLTK punkt tokenizer")
                return True
            except (LookupError, Exception):
                return False
    
    def evaluate(self, reference: str, input_text: str) -> Dict[str, Any]:
        """
        Evaluate an input text using ROUGE score.
        
        Args:
            reference: Reference text
            input_text: Input text to evaluate
            
        Returns:
            Dictionary of ROUGE score results
        """

        # Clean reference and input text before evaluation
        reference = self._clean_text(reference)
        input_text = self._clean_text(input_text)

        # Check if NLTK resources are available
        if not self.nltk_resources_available:
            return {
                'evaluator_type': 'rouge',
                'error': "NLTK resources are not available. Run 'python download_nltk_data.py' to install them."
            }
            
        # Check if rouge_score is available
        if not self.rouge_score_available:
            # Use fallback implementation
            return self._fallback_rouge(reference, input_text)
        
        try:
            # Import rouge-score here to avoid loading it during initialization
            from rouge_score import rouge_scorer
            
            # Tokenize the texts into sentences
            reference_sentences_raw = self._tokenize_into_sentences(reference)
            input_sentences_raw = self._tokenize_into_sentences(input_text)

            # Remove stopwords (used only for scoring)
            reference_sentences_clean = [self.remove_stopwords_rouge(sent) for sent in reference_sentences_raw]
            input_sentences_clean = [self.remove_stopwords_rouge(sent) for sent in input_sentences_raw]
            
            # Set up the ROUGE scorer
            scorer = rouge_scorer.RougeScorer(
                self.rouge_types, 
                use_stemmer=self.use_stemmer
            )

            # Get rid of stopwords for the entire text
            reference = self.remove_stopwords_rouge(reference)
            input_text = self.remove_stopwords_rouge(input_text)
            
            # Calculate ROUGE for the entire text
            overall_scores = scorer.score(reference, input_text)
            
            # Convert scores to a more JSON-friendly format
            overall_dict = {}
            for rouge_type, score_obj in overall_scores.items():
                overall_dict[rouge_type] = {
                    'precision': score_obj.precision,
                    'recall': score_obj.recall,
                    'fmeasure': score_obj.fmeasure
                }
            
            # Calculate sentence-level ROUGE scores
            sentence_rouge_scores = []
            
            for i, input_sent_clean in enumerate(input_sentences_clean):
                input_sent_raw = input_sentences_raw[i]
                
                best_score = {rouge_type: {'fmeasure': 0, 'precision': 0, 'recall': 0} for rouge_type in self.rouge_types}
                best_reference = {rouge_type: None for rouge_type in self.rouge_types}
                best_reference_raw = {rouge_type: None for rouge_type in self.rouge_types}
                
                for j, reference_sent_clean in enumerate(reference_sentences_clean):
                    reference_sent_raw = reference_sentences_raw[j]
                    sent_scores = scorer.score(reference_sent_clean, input_sent_clean)

                    for rouge_type in self.rouge_types:
                        sent_score = sent_scores[rouge_type]
                        if sent_score.fmeasure > best_score[rouge_type]['fmeasure']:
                            best_score[rouge_type] = {
                                'fmeasure': sent_score.fmeasure,
                                'precision': sent_score.precision,
                                'recall': sent_score.recall
                            }
                            best_reference[rouge_type] = reference_sent_clean
                            best_reference_raw[rouge_type] = reference_sent_raw


                # Store the best match and ROUGE scores for each input sentence
                sentence_rouge_scores.append({
                    'input': input_sent_raw,
                    'best_reference': best_reference_raw,
                    'rouge_for_this_sentence': best_score
                })

            # Prepare and return results
            results = {
                'evaluator_type': 'rouge',
                'rouge_types': self.rouge_types,
                'use_stemmer': self.use_stemmer,
                'scores': {
                    'overall': overall_dict,
                    'sentence_level': sentence_rouge_scores
                }
            }
            
            return results
            
        except Exception as e:
            print(f"Error calculating ROUGE score: {e}")
            # Return error information
            return {
                'evaluator_type': 'rouge',
                'error': str(e)
            }
        
    def _tokenize_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # try:
        #     return nltk.sent_tokenize(text)
        # except Exception as e:
        #     # Fallback tokenization
        #     print(f"Warning: NLTK sentence tokenization failed: {e}")
        #     return [s.strip() for s in text.split('.') if s.strip()]
        return [line.strip() for line in text.strip().split('\n') if line.strip()]

    def _fallback_rouge(self, reference: str, input_text: str) -> Dict[str, Any]:
        """
        A simple fallback implementation of ROUGE using NLTK.
        This is only used if rouge_score package is not available.
        
        Args:
            reference: Reference text
            input_text: Input text to evaluate
            
        Returns:
            Dictionary of ROUGE-like scores
        """
        try:
            # Tokenize into words
            reference_tokens = set(nltk.word_tokenize(reference.lower()))
            input_tokens = set(nltk.word_tokenize(input_text.lower()))
            
            # Calculate simple unigram overlap
            if not reference_tokens:
                precision = 1.0 if not input_tokens else 0.0
                recall = 1.0
            elif not input_tokens:
                precision = 0.0
                recall = 0.0
            else:
                common_tokens = reference_tokens.intersection(input_tokens)
                precision = len(common_tokens) / len(input_tokens) if input_tokens else 0.0
                recall = len(common_tokens) / len(reference_tokens) if reference_tokens else 0.0
            
            # Calculate F1 score
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
            
            return {
                'evaluator_type': 'rouge',
                'implementation': 'fallback',
                'scores': {
                    'overall': {
                        'simple_rouge1': {
                            'precision': precision,
                            'recall': recall,
                            'fmeasure': f1
                        }
                    }
                },
                'warning': "Using fallback ROUGE implementation. Install rouge-score for better results."
            }
        except Exception as e:
            print(f"Error in fallback ROUGE calculation: {e}")
            return {
                'evaluator_type': 'rouge',
                'error': f"Fallback ROUGE calculation failed: {e}"
            }
