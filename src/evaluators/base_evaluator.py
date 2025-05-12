from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseEvaluator(ABC):
    """Base class for all evaluators."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with configuration.
        
        Args:
            config: Full configuration dictionary
        """
        self.config = config

        self.stopwords = set([
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
            "you", "your", "yours", "yourself", "yourselves", "he", "him",
            "his", "himself", "she", "her", "hers", "herself", "it", "its",
            "itself", "they", "them", "their", "theirs", "themselves",
            "what", "which", "who", "whom", "this", "that", "these", "those",
            "am", "is", "are", "was", "were", "be", "been", "being", "have",
            "has", "had", "having", "do", "does", "did", "doing", "a", "an",
            "the", "and", "but", "if", "or", "because", "as", "until", "while",
            "of", "at", "by", "for", "with", "about", "against", "between",
            "into", "through", "during", "before", "after", "above", "below",
            "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
            "again", "further", "then", "once", "here", "there", "when", "where",
            "why", "how", "all", "any", "both", "each", "few", "more", "most",
            "other", "some", "such", "no", "nor", "not", "only", "own", "same",
            "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
            "should", "now"
        ])

    def remove_stopwords_bleu(self, tokens):
        """Remove stopwords from a list of tokens."""
        return [t for t in tokens if t.lower() not in self.stopwords]
    
    def remove_stopwords_rouge(self, text: str) -> str:
        """Remove stopwords from a given text."""
        tokens = text.split()
        filtered_tokens = [token for token in tokens if token.lower() not in self.stopwords]
        return ' '.join(filtered_tokens)
    
    @abstractmethod
    def evaluate(self, reference: str, input_text: str) -> Dict[str, Any]:
        """
        Evaluate an input text against a reference text.
        
        Args:
            reference: Reference text
            input_text: Input text to evaluate
            
        Returns:
            Dictionary of evaluation results
        """
        pass

    # Preprocessing text by removing initial lines and threat tags - STRIDEGPT tool
    def _clean_text(self, text: str) -> str:
        """
        Remove the first two lines, STRIDE labels, and adjusts the final punctuation.

        Rules:
        - Removes STRIDE labels (e.g., | Tampering |)
        - Replaces the last '|' of the line with a period
        - Removes any remaining pipes '|'

        Args:
            text: The text to be cleaned.

        Returns:
            Cleaned text.
        """
        lines = text.split('\n')
        lines = lines[2:]  # Remove the first two lines

        threat_labels = [
            "Spoofing", "Tampering", "Repudiation",
            "Information Disclosure", "Denial of Service",
            "Elevation of Privilege"
        ]

        cleaned_lines = []
        for line in lines:
            # Remove STRIDE labels with or without spaces
            for label in threat_labels:
                for pattern in [
                    f"| {label} |", f"|{label} |", f"| {label}|", f"|{label}|"
                ]:
                    line = line.replace(pattern, "")

            # If there's at least one pipe, treat the last as end of sentence
            if '|' in line:
                parts = line.rsplit('|', 1)
                before_last = parts[0].replace('|', '').strip()
                after_last = parts[1].strip().lstrip('.').strip()

                if not before_last.endswith('.'):
                    before_last += '.'

                line = f"{before_last} {after_last}"
            else:
                line = line.strip()

            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)
