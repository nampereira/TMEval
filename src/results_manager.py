import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import defaultdict
import uuid

class ResultsManager:
    """Handles aggregation, formatting, and persistent storage of evaluation results."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ResultsManager with the given configuration.

        Args:
            config (Dict[str, Any]): Configuration parameters including output settings and evaluator types.
        """
        self.config = config
        self.output_format = config.get('output', {}).get('format', 'json')
        self.output_directory = config.get('output', {}).get('directory', 'results')
        self.filename_template = config.get('output', {}).get('filename_template', '{input_filename}_eval.json')

        # Retrieve evaluator types from configuration, defaulting to single 'dimension' type if not specified
        evaluator_types = config.get('evaluator', {}).get('types', [])
        if not evaluator_types:
            evaluator_types = [config.get('evaluator', {}).get('type', 'dimension')]

        self.evaluator_types = evaluator_types
        self.is_multi_evaluator = len(evaluator_types) > 1

        # Generate a unique identifier for this run instance to track result files
        self.run_id = uuid.uuid4().hex

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
        Persist evaluation results to disk with enriched metadata.

        Args:
            results (Dict[str, Any]): The evaluation results data structure.
            reference (str): Reference text used for evaluation.
            input_text (str): The input text evaluated.
            reference_filename (Optional[str]): Optional filename of the reference input.
            input_filename (Optional[str]): Optional filename of the evaluated input.
            title (str): Optional descriptive title for the evaluation pair.
            description (str): Optional description providing context for the evaluation pair.
            tags (Optional[List[str]]): Optional list of tags categorizing the evaluation.

        Returns:
            str: The full file path where results were saved.
        """
        # Ensure tags list is initialized
        if tags is None:
            tags = []

        # Compose the complete result payload including metadata and timestamps
        result_data = {
            'run_id': self.run_id,
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

        # Calculate weighted average score across all dimensions, storing averages per dimension
        dimensions = results.get('dimensions', {})
        weighted_average = 0
        for dim, dim_data in dimensions.items():
            responses = dim_data.get('responses', {})
            total_score = 0
            for key in responses:
                # Extract numeric value; handles cases like "Response: <int>"
                value_str = key.split(":", 1)[-1].strip() if ":" in key else key
                total_score += int(value_str)
            average_score = total_score / len(responses) if responses else 0
            dimensions[dim]['average'] = average_score
            weighted_average += average_score * dim_data.get('weight', 0)
        dimensions['score'] = weighted_average

        # Ensure output directory exists
        os.makedirs(self.output_directory, exist_ok=True)

        # Construct output filename incorporating evaluator type(s), run ID, and timestamp
        if input_filename:
            base_name = os.path.splitext(os.path.basename(input_filename))[0]

            if self.is_multi_evaluator:
                evaluator_suffix = "_multi"
            elif 'evaluator_type' in results:
                evaluator_suffix = f"_{results['evaluator_type']}"
            else:
                evaluator_suffix = f"_{self.evaluator_types[0]}"

            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = self.filename_template.format(
                input_filename=f"{base_name}{evaluator_suffix}_{self.run_id}_{timestamp_str}"
            )
        else:
            # Filename for inputs without source file, based on evaluator and timestamp
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            if self.is_multi_evaluator:
                output_filename = f"eval_multi_{self.run_id}_{timestamp_str}.json"
            elif 'evaluator_type' in results:
                output_filename = f"eval_{results['evaluator_type']}_{self.run_id}_{timestamp_str}.json"
            else:
                output_filename = f"eval_{self.evaluator_types[0]}_{self.run_id}_{timestamp_str}.json"

        output_path = os.path.join(self.output_directory, output_filename)

        # Serialize results to JSON file; extend here for additional formats if necessary
        if self.output_format == 'json':
            with open(output_path, 'w') as f:
                json.dump(result_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported output format: {self.output_format}")

        return output_path

    def load_all_results(self, evaluator_type: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load and group all result files filtered by evaluator type, organizing by run ID.

        Args:
            evaluator_type (str): Evaluator type to filter relevant result files.

        Returns:
            Dict[str, List[Dict[str, Any]]]: Mapping from run_id to a list of corresponding results.
        """
        results_by_run_id = defaultdict(list)
        if not os.path.exists(self.output_directory):
            return results_by_run_id

        # Iterate through stored JSON result files matching the evaluator type filter
        for filename in os.listdir(self.output_directory):
            if filename.endswith('.json') and evaluator_type in filename:
                full_path = os.path.join(self.output_directory, filename)
                try:
                    with open(full_path, 'r') as f:
                        data = json.load(f)
                        run_id = data.get('run_id', 'unknown_run')
                        results_by_run_id[run_id].append(data)
                except Exception as e:
                    print(f"Warning: Unable to load file {full_path}: {e}")

        # Define a fixed custom sorting order for result titles
        custom_order = [
            "Spoofing",
            "Tampering",
            "Repudiation",
            "Information Disclosure",
            "Denial of Service",
            "Elevation of Privilege"
        ]

        # Map titles to their position in custom order for sorting purposes
        order_map = {title.upper(): index for index, title in enumerate(custom_order)}

        def get_sort_key(result: Dict[str, Any]) -> int:
            title = result.get('metadata', {}).get('title', '').upper()
            return order_map.get(title, len(custom_order))

        # Sort the list of results per run_id according to the custom title order
        for run_id, results_list in results_by_run_id.items():
            results_list.sort(key=get_sort_key)

        # Return dictionary sorted by run_id keys for consistent ordering
        return dict(sorted(results_by_run_id.items()))
