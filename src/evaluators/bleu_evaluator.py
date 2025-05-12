import os
from typing import Dict, Any, List, Tuple
from .base_evaluator import BaseEvaluator
import nltk
import math

class BLEUEvaluator(BaseEvaluator):
    """Evaluates input texts using BLEU score metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with configuration.
        
        Args:
            config: Full configuration dictionary
        """
        super().__init__(config)
        self.bleu_config = config.get('evaluator', {}).get('bleu', {})
        self.max_ngram = self.bleu_config.get('max_ngram', 4)
        self.weights = self.bleu_config.get('weights', None)
        
        # Set up default weights if not specified
        if self.weights is None:
            # Default: equal weights for n-grams up to max_ngram
            self.weights = [1.0/self.max_ngram] * self.max_ngram
        
        # Print initialization info
        print(f"Initializing BLEU evaluator with max_ngram={self.max_ngram}")
        print(f"Using weights: {self.weights}")
        
        # Check if NLTK resources are available
        self.nltk_resources_available = self._check_nltk_resources()
        
        if not self.nltk_resources_available:
            print("\n==============================================================================")
            print("BLEU evaluator requires NLTK data resources that are not installed.")
            print("Please download them using:")
            print("  python download_nltk_data.py")
            print("  or")
            print("  import nltk; nltk.download('punkt')")
            print("==============================================================================\n")
    
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
        Evaluate an input text using BLEU score.
        
        Args:
            reference: Reference text
            input_text: Input text to evaluate
            
        Returns:
            Dictionary of BLEU score results
        """
        # Check if NLTK resources are available
        if not self.nltk_resources_available:
            return {
                'evaluator_type': 'bleu',
                'error': "NLTK resources are not available. Run 'python download_nltk_data.py' to install them."
            }
            
        try:
            # Clean the reference and input texts.
            reference_cleaned = self._clean_text(reference)
            input_cleaned = self._clean_text(input_text)

            # Tokenize the texts into sentences and words
            reference_sentences = self._tokenize_into_sentences(reference_cleaned)
            input_sentences = self._tokenize_into_sentences(input_cleaned)
            
            # Tokenize each sentence into words
            reference_tokens = [self._tokenize_into_words(sent) for sent in reference_sentences]
            input_tokens = [self._tokenize_into_words(sent) for sent in input_sentences]
            
            # Get rid of stopwords
            input_tokens = [self.remove_stopwords_bleu(tokens) for tokens in input_tokens]
            reference_tokens = [self.remove_stopwords_bleu(tokens) for tokens in reference_tokens]
            
            # Flatten reference sentences for corpus-level BLEU
            reference_tokens_flat = [token for sent in reference_tokens for token in sent]
            input_tokens_flat = [token for sent in input_tokens for token in sent]
            
            # Calculate corpus-level BLEU score
            corpus_bleu = self._calculate_bleu(
                references=[reference_tokens_flat],
                hypothesis=input_tokens_flat
            )
            
            # Calculate BLEU for individual n-gram levels
            ngram_scores = {}
            for n in range(1, self.max_ngram + 1):
                weights = [0] * self.max_ngram
                weights[n-1] = 1.0
                score = self._calculate_bleu(
                    references=[reference_tokens_flat],
                    hypothesis=input_tokens_flat,
                    weights=weights
                )
                ngram_scores[f"{n}-gram"] = score
            
            # Calculate sentence-level BLEU scores
            sentence_bleu_scores = []
            for i, input_sent_tokens in enumerate(input_tokens):
                best_score = 0 
                best_reference = None 

                # For each sentence in the reference calculate the BLEU score
                for reference_sent_tokens in reference_tokens:
                    score = self._calculate_bleu(
                        references=[reference_sent_tokens],
                        hypothesis=input_sent_tokens
                    )
                    
                    # If the BLEU score is better update the best match.
                    if score > best_score:
                        best_score = score
                        best_reference = reference_sentences[reference_tokens.index(reference_sent_tokens)]
                
                sentence_bleu_scores.append({
                    'input': input_sentences[i],
                    'best_reference': best_reference,
                    'bleu_for_this_sentence': best_score
                })
            
            # Prepare and return results
            results = {
                'evaluator_type': 'bleu',
                'max_ngram': self.max_ngram,
                'weights': self.weights,
                'scores': {
                    'overall': corpus_bleu,
                    'ngram_scores': ngram_scores,
                    'sentence_level': sentence_bleu_scores
                }
            }
            
            return results
            
        except Exception as e:
            print(f"Error calculating BLEU score: {e}")
            # Return error information
            return {
                'evaluator_type': 'bleu',
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
    
    def _tokenize_into_words(self, sentence: str) -> List[str]:
        """
        Tokenize a sentence into words.
        
        Args:
            sentence: Sentence to tokenize
            
        Returns:
            List of words/tokens
        """
        try:
            return nltk.word_tokenize(sentence.lower())
        except Exception as e:
            # Fallback tokenization
            print(f"Warning: NLTK word tokenization failed: {e}")
            return [w.strip().lower() for w in sentence.split() if w.strip()]
    
    def _calculate_bleu(self, references: List[List[str]], hypothesis: List[str], 
                       weights: List[float] = None) -> float:
        """
        Calculate BLEU score.
        
        Args:
            references: List of reference token lists
            hypothesis: Hypothesis token list
            weights: Weights for n-grams (default: equal weights for n-grams up to max_ngram)
            
        Returns:
            BLEU score
        """
        weights = weights or self.weights
        
        # Use NLTK's BLEU implementation
        try:
            # If hypothesis is empty, return 0
            if not hypothesis:
                return 0.0
                
            # If all references are empty, return 0 if hypothesis is empty, 1 otherwise
            if all(not ref for ref in references):
                return 1.0 if not hypothesis else 0.0
                
            # Calculate BLEU score
            bleu = nltk.translate.bleu_score.sentence_bleu(
                references, 
                hypothesis,
                weights=weights,
                smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1
            )
            
            return bleu
        except Exception as e:
            print(f"Error in BLEU calculation: {e}")
            return 0.0




