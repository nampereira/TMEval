import os
from typing import Dict, Any, List, Tuple, Optional

class FileProcessor:
    """Handles reading reference and input files based on configuration."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with configuration.
        
        Args:
            config: Full configuration dictionary
        """
        self.config = config
        
        # New configuration method (explicit file pairs)
        self.file_pairs = config.get('files', [])
        
        # Legacy configuration method (directory scanning)
        self.reference_dir = config.get('input', {}).get('reference_dir', 'input/references')
        self.input_dir = config.get('input', {}).get('input_dir', 'input/inputs')
        self.file_extension = config.get('input', {}).get('file_extension', '.txt')
        
        # Flag to track which method we're using
        self.using_explicit_files = len(self.file_pairs) > 0
    
    def get_file_pairs(self) -> List[Dict[str, Any]]:
        """
        Get list of file pairs to process, either from explicit configuration
        or by scanning directories.
        
        Returns:
            List of dictionaries, each containing file paths and metadata
        """
        if self.using_explicit_files:
            return self._get_explicit_file_pairs()
        else:
            return self._get_directory_file_pairs()
    
    def _get_explicit_file_pairs(self) -> List[Dict[str, Any]]:
        """
        Get file pairs explicitly defined in configuration.
        
        Returns:
            List of dictionaries with file paths and metadata
        """
        validated_pairs = []
        
        for pair in self.file_pairs:
            reference_file = pair.get('reference_file')
            input_file = pair.get('input_file')
            
            # Validate that both files exist
            if not reference_file or not input_file:
                print(f"Warning: Skipping incomplete file pair: {pair}")
                continue
                
            if not os.path.exists(reference_file):
                print(f"Warning: Reference file does not exist: {reference_file}")
                continue
                
            if not os.path.exists(input_file):
                print(f"Warning: Input file does not exist: {input_file}")
                continue
            
            # Add the validated pair with all metadata
            validated_pair = {
                'reference_file': reference_file,
                'input_file': input_file,
                'title': pair.get('title', ''),
                'description': pair.get('description', ''),
                'tags': pair.get('tags', []),
                'base_filename': os.path.splitext(os.path.basename(input_file))[0]
            }
            
            validated_pairs.append(validated_pair)
        
        return validated_pairs
    
    def _get_directory_file_pairs(self) -> List[Dict[str, Any]]:
        """
        Find pairs of reference and input files based on matching filenames in directories.
        
        Returns:
            List of dictionaries, each containing reference and input file paths
        """
        file_pairs = []
        
        # Get all input files
        input_files = self._get_files_in_directory(self.input_dir)
        
        for input_file in input_files:
            # Extract the base filename (without extension)
            base_filename = os.path.splitext(os.path.basename(input_file))[0]
            
            # Look for a matching reference file
            reference_file = self._find_matching_reference_file(base_filename)
            
            if reference_file:
                file_pairs.append({
                    'reference_file': reference_file,
                    'input_file': input_file,
                    'base_filename': base_filename,
                    'title': '',  # Empty metadata for directory-based pairs
                    'description': '',
                    'tags': []
                })
        
        return file_pairs
    
    def read_file(self, file_path: str) -> Tuple[str, str]:
        """
        Read file content and return it along with the filename.
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            Tuple of (file content, filename)
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return content, os.path.basename(file_path)
    
    def _get_files_in_directory(self, directory: str) -> List[str]:
        """
        Get all files with the specified extension in a directory.
        
        Args:
            directory: Directory to search
            
        Returns:
            List of file paths
        """
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            return []
        
        files = []
        for filename in os.listdir(directory):
            if filename.endswith(self.file_extension):
                files.append(os.path.join(directory, filename))
        
        return files
    
    def _find_matching_reference_file(self, base_filename: str) -> Optional[str]:
        """
        Find a reference file that matches the base filename.
        
        Args:
            base_filename: Base filename to search for
            
        Returns:
            Path to the matching reference file, or None if not found
        """
        reference_file_path = os.path.join(self.reference_dir, f"{base_filename}{self.file_extension}")
        
        if os.path.exists(reference_file_path):
            return reference_file_path
        
        return None
