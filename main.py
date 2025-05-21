import os
import argparse
from src.config_parser import load_config
from src.evaluators import get_evaluator
from src.results_manager import ResultsManager
from src.file_processor import FileProcessor
from dotenv import load_dotenv
# Dashboards
from src.dashboards.bleu_dashboard import launch_bleu_dashboard
from src.dashboards.rouge_dashboard import launch_rouge_dashboard
from src.dashboards.dimension_dashboard import launch_dimension_dashboard
from src.dashboards.bertscore_dashboard import launch_bertscore_dashboard

def main():
    """Main entry point for the evaluation script."""
    # Load environment variables
    load_dotenv()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Evaluate input texts using various metrics')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--reference-file', type=str, help='Path to specific reference text file')
    parser.add_argument('--input-file', type=str, help='Path to specific input text file')
    parser.add_argument('--reference', type=str, help='Reference text (direct input)')
    parser.add_argument('--input', type=str, help='Input text (direct input)')
    parser.add_argument('--reference-dir', type=str, help='Directory containing reference files (overrides config)')
    parser.add_argument('--input-dir', type=str, help='Directory containing input files (overrides config)')
    parser.add_argument('--llm', type=str, help='Specific LLM to use (overrides config)')
    parser.add_argument('--evaluator', type=str, help='Single evaluator type to use (overrides config)')
    parser.add_argument('--evaluators', type=str, help='Comma-separated list of evaluator types to use (overrides config)')
    parser.add_argument('--title', type=str, help='Title for the text pair (for command-line input)')
    parser.add_argument('--description', type=str, help='Description for the text pair (for command-line input)')
    parser.add_argument('--num-completions', type=int, help='Number of completions to generate per prompt')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments if provided
    if args.reference_dir:
        config['input']['reference_dir'] = args.reference_dir
    if args.input_dir:
        config['input']['input_dir'] = args.input_dir
    if args.llm:
        if args.llm in config.get('llms', {}):
            config['active_llm'] = args.llm
        else:
            print(f"Warning: LLM '{args.llm}' is not configured. Using the default LLM instead.")
    
    # Handle evaluator type overrides
    if args.evaluators:
        # Multiple evaluators specified as comma-separated list
        evaluator_types = [e.strip() for e in args.evaluators.split(',')]
        config['evaluator']['types'] = evaluator_types
        print(f"Using evaluators: {', '.join(evaluator_types)}")
    elif args.evaluator:
        # Single evaluator specified
        config['evaluator']['types'] = [args.evaluator]
        print(f"Using evaluator: {args.evaluator}")
    
    if args.num_completions:
        active_llm = config.get('active_llm', 'claude')
        if active_llm in config.get('llms', {}):
            config['llms'][active_llm]['num_completions'] = args.num_completions
            print(f"Set number of completions for {active_llm} to {args.num_completions}")
    
    # Initialize evaluator and results manager
    evaluator = get_evaluator(config)
    results_manager = ResultsManager(config)
    
    # Process single files or text if provided
    if args.reference_file and args.input_file:
        title = args.title or ""
        description = args.description or ""
        process_single_pair(args.reference_file, args.input_file, evaluator, results_manager, title, description)
    elif args.reference and args.input:
        title = args.title or ""
        description = args.description or ""
        process_direct_input(args.reference, args.input, evaluator, results_manager, title, description)
    else:
        # Process files based on configuration
        process_configured_files(config, evaluator, results_manager)

def process_single_pair(reference_file, input_file, evaluator, results_manager, title="", description="", tags=None):
    """Process a single pair of reference and input files."""
    if tags is None:
        tags = []
        
    with open(reference_file, 'r', encoding='utf-8') as f:
        reference = f.read()
    
    with open(input_file, 'r', encoding='utf-8') as f:
        input_text = f.read()
    
    reference_filename = os.path.basename(reference_file)
    input_filename = os.path.basename(input_file)
    
    print(f"Evaluating files: {reference_filename} and {input_filename}")
    results = evaluator.evaluate(reference, input_text)
    output_path = results_manager.save_results(
        results, reference, input_text, reference_filename, input_filename,
        title, description, tags
    )
    print(f"Results saved to {output_path}")

def process_direct_input(reference, input_text, evaluator, results_manager, title="", description="", tags=None):
    """Process direct text input."""
    if tags is None:
        tags = []
        
    print("Evaluating direct text input")
    results = evaluator.evaluate(reference, input_text)
    output_path = results_manager.save_results(
        results, reference, input_text, None, None,
        title, description, tags
    )
    print(f"Results saved to {output_path}")

def process_configured_files(config, evaluator, results_manager):
    """Process file pairs based on configuration."""
    file_processor = FileProcessor(config)
    file_pairs = file_processor.get_file_pairs()
    
    if not file_pairs:
        print("No matching reference and input files found.")
        return
    
    print(f"Found {len(file_pairs)} reference-input pairs to evaluate.")
    
    # Display info about evaluators
    if hasattr(evaluator, 'evaluator_types'):
        print(f"Using multiple evaluators: {', '.join(evaluator.evaluator_types)}")
    else:
        evaluator_type = evaluator.__class__.__name__.replace('Evaluator', '').lower()
        print(f"Using evaluator: {evaluator_type}")
    
    if 'DimensionEvaluator' in str(evaluator.__class__) or hasattr(evaluator, 'evaluators'):
        active_llm = config.get('active_llm', 'claude')
        num_completions = config.get('llms', {}).get(active_llm, {}).get('num_completions', 1)
        print(f"Using LLM: {active_llm} with {num_completions} completion(s) per prompt")
    
    all_results = []
    
    for pair in file_pairs:
        reference_file = pair['reference_file']
        input_file = pair['input_file']
        
        reference, reference_filename = file_processor.read_file(reference_file)
        input_text, input_filename = file_processor.read_file(input_file)
        
        title = pair.get('title', '')
        description = pair.get('description', '')
        tags = pair.get('tags', [])
        
        print(f"\nEvaluating: {input_filename}")
        if title:
            print(f"Title: {title}")
        
        results = evaluator.evaluate(reference, input_text)
        output_path = results_manager.save_results(
            results, reference, input_text, reference_filename, input_filename,
            title, description, tags
        )
        print(f"Results saved to {output_path}")
        
        # Add title to use in the dashboard
        results['title'] = title or os.path.splitext(input_filename)[0]
        all_results.append(results)
    
    # Launch the dashboard according to the evaluator
    evaluator_type = evaluator.__class__.__name__.replace('Evaluator', '').lower()
    if evaluator_type == 'bleu':
        all_results = results_manager.load_all_results(evaluator_type)
        launch_bleu_dashboard(all_results)
    elif evaluator_type == 'rouge':
        all_results = results_manager.load_all_results(evaluator_type)
        launch_rouge_dashboard(all_results)
    elif evaluator_type == 'dimension':
        all_results = results_manager.load_all_results(evaluator_type)
        launch_dimension_dashboard(all_results)
    elif evaluator_type == 'bertscore':
        all_results = results_manager.load_all_results(evaluator_type)
        launch_bertscore_dashboard(all_results)
    else:
        print(f"No dashboard available for evaluator type: {evaluator_type}")

if __name__ == '__main__':
    main()
