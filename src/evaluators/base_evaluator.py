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
