# Threat Model Evaluation Tool (TMEval)

TMEval is a modular Python application that evaluates text using multiple evaluation strategies. It leverages Large Language Models (LLMs) like Claude, ChatGPT, or Gemini both for subjective dimension-based evaluation and for semantic similarity assessment similar to BERTScore.

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API keys** in a `.env` file:
   ```
   CLAUDE_API_KEY=your_claude_api_key
   OPENAI_API_KEY=your_openai_api_key
   GEMINI_API_KEY=your_gemini_api_key
   ```

3. **Configure file pairs** in `config.yaml`:
   ```yaml
   files:
     - reference_file: "input/references/dos.txt"
       input_file: "input/inputs/dos.txt"
       title: "Denial of Service"
       description: "Denial of Service Threats"
   ```

4. Use the Dependency Installer
```bash
python install_dependencies.py
```

5. **Run the evaluation**:
   ```bash
   python main.py
   ```

6. **Check results** in the `results/` directory

## Application Details

### Purpose

TMEval provides multiple ways to evaluate text:

#### Dimension-based Evaluation (LLM-as-a-Judge)
This is inpired on [G-EVAL](https://arxiv.org/pdf/2303.16634) and uses an LLM to evaluate different dimensions of the generated input against a reference.

Each dimension is implemented by a specific prompt (see `prompts` folder). It currently has the following dimensions:

- **Relevance**: How well the input text captures important content from the reference
- **Consistency**: Factual alignment between the input text and reference text
- **Fluency**: Quality of individual sentences in the input text
- **Non-Redundancy**: No unnecessary repetition.

- **TODO: Analyse results in detail**

#### LLM-based BERTScore Evaluation
- Uses an LLM to assess semantic similarity between reference and input texts
- Provides precision, recall, and F1 scores similar to traditional BERTScore
- Leverages the LLM's understanding of semantic meaning rather than token embeddings

- **TODO: analyse results; check sentence splitting; consider sorting reference and input ?**

#### BLEU and ROUGE

More traditional BLEU and ROUGE metrics evaluators are also included. 

- **TODO: Their results and relevance needs to be evaluated.**

#### Metrics Details

Additional details in [README-metrics.md](README-metrics.md)

### Key Features

- **Multiple evaluator types**: Run multiple evaluators in a single execution
- **Configurable LLM selection**: Choose which LLM to use for evaluations
- **Multiple completions**: Generate and store multiple responses per prompt
- **File pair configuration**: Define specific file pairs with metadata in the config
- **Flexible configuration**: Easily modify prompts, weights, and settings
- **Comprehensive output**: Results include original texts, metadata, and evaluations

## Application Structure

```
tmeval/
├── config.yaml                # Configuration file
├── main.py                    # Main entry point
├── requirements.txt           # Dependencies
├── prompts/                   # Prompt template files
│   ├── relevance.txt          # Relevance evaluation prompt
│   ├── consistency.txt        # Consistency evaluation prompt
│   ├── fluency.txt            # Fluency evaluation prompt
│   ├── coherence.txt          # Coherence evaluation prompt
│   └── bertscore.txt          # BERTScore evaluation prompt
├── input/                     # Input directories
│   ├── references/            # Reference texts
│   └── inputs/                # Input texts to evaluate
├── results/                   # Evaluation results
└── src/                       # Source code
    ├── __init__.py
    ├── config_parser.py       # Configuration loading
    ├── evaluators/            # Evaluator implementations
    │   ├── __init__.py        # Evaluator factory
    │   ├── base_evaluator.py  # Base evaluator interface
    │   ├── dimension_evaluator.py  # LLM-based evaluator
    │   ├── bertscore_evaluator.py  # BERTScore-like evaluator
    │   └── multi_evaluator.py  # Multi-evaluator implementation
    ├── file_processor.py      # File handling
    ├── results_manager.py     # Results aggregation and saving
    └── llm_apis/              # LLM API integrations
        ├── __init__.py
        ├── base.py            # Base LLM API class
        ├── claude.py          # Claude-specific implementation
        ├── chatgpt.py         # ChatGPT-specific implementation
        └── gemini.py          # Gemini-specific implementation
```

## Configuration Options

TMEval is configured through `config.yaml`:

### File Pair Configuration

```yaml
# Input file pairs and metadata
files:
  - reference_file: "input/references/article1.txt"
    input_file: "input/inputs/article1.txt"
    title: "Climate Change Overview"
    description: "A summary of climate change causes and effects"
    tags: ["science", "environment", "climate"]
```

### Multiple Evaluators Configuration

```yaml
# Evaluator configuration with multiple evaluator types
evaluator:
  types: 
    - "dimension"  # LLM-based dimension evaluation
    - "bertscore"  # LLM-based semantic similarity evaluation
  
  # BERTScore-specific configuration
  bertscore:
    implementation: "llm"  # Uses LLM instead of traditional BERTScore
```

### LLM Configuration

```yaml
# Available LLM configurations
llms:
  claude:
    api_key: ${CLAUDE_API_KEY}
    model: "claude-3-opus-20240229"
    api_url: "https://api.anthropic.com/v1/messages"
    num_completions: 3  # Generate 3 completions per prompt
  # Other LLMs...

# Specify which LLM to use (must match one of the keys in the llms section)
active_llm: "claude"
```

### Evaluation Dimensions (for dimension-based evaluator)

```yaml
dimensions:
  relevance:
    prompt_file: "prompts/relevance.txt"
    weight: 0.25
  # Other dimensions...
```

### Output Settings

```yaml
output:
  format: "json"
  directory: "results"
  filename_template: "{input_filename}_eval.json"
```

## Usage Examples

### Process File Pairs from Configuration

```bash
python main.py
```

### Evaluate Specific Files

```bash
python main.py --reference-file input/references/article1.txt --input-file input/inputs/article1.txt --title "Custom Title" --description "Custom description"
```

### Direct Text Input

```bash
python main.py --reference "Reference text" --input "Input text to evaluate" --title "Custom Example"
```

### Specify a Single Evaluator

```bash
python main.py --evaluator bertscore
```

### Specify Multiple Evaluators

```bash
python main.py --evaluators dimension,bertscore
```

### Generate Multiple Completions

```bash
python main.py --num-completions 5
```
### Configuration

```yaml
evaluator:
  bertscore:
    implementation: "llm"  # Uses LLM instead of traditional BERTScore
```

## Multiple Evaluators

TMEval supports running multiple evaluator types in sequence on the same input. This feature allows you to:

- Compare different evaluation approaches
- Get a more comprehensive assessment of text quality
- Combine subjective and objective evaluations

To configure multiple evaluators:

1. **In the configuration file**:
   ```yaml
   evaluator:
     types: 
       - "dimension"  # LLM-based dimension evaluation
       - "bertscore"  # LLM-based semantic similarity evaluation
   ```

2. **From the command line**:
   ```bash
   python main.py --evaluators dimension,bertscore
   ```

The results from all evaluators are combined into a single output file with each evaluator's results stored in its own section.

### Multi-evaluator Result Format

When using multiple evaluators, the results are structured like this:

```json
{
  "timestamp": "2023-04-30T12:34:56.789012",
  "reference": { /* reference text info */ },
  "input": { /* input text info */ },
  "metadata": { /* metadata info */ },
  "results": {
    "evaluator_type": "multi",
    "evaluators_used": ["DimensionEvaluator", "BERTScoreEvaluator"],
    "results": {
      "dimension": {
        /* dimension evaluator results */
      },
      "bertscore": {
        /* bertscore evaluator results */
      }
    }
  }
}
```

## Multiple Completions

TMEval supports generating multiple completions (different responses) from each LLM for the same prompt. This feature is useful for:

- Assessing the variability in LLM responses
- Collecting a broader range of evaluations
- Increasing confidence in the evaluation results

To configure multiple completions:

1. **In the configuration file**:
   ```yaml
   llms:
     claude:
       # ... other settings
       num_completions: 3  # Generate 3 different completions
   ```

2. **From the command line**:
   ```bash
   python main.py --num-completions 5
   ```

The number of completions that can be generated varies by LLM:
- **ChatGPT**: Natively supports multiple completions in a single API call
- **Claude**: Generates multiple completions through separate API calls with different system messages
- **Gemini**: Uses a combination of native multiple completion support (when available) and temperature variation

## File Format

### Reference and Input Files

These are plain text files containing the reference text and the input text to evaluate. File pairs should have the same base filename for directory-based scanning to work.

### Metadata Options

For each file pair, you can specify:
- **title**: A descriptive title for the text pair
- **description**: A detailed description of the text pair
- **tags**: A list of keywords or categories

This metadata is stored alongside the evaluation results and can be used for filtering, searching, or organizing results.

## Result Format

Results are saved as JSON files with a structure appropriate to the evaluator type used.

### Dimension Evaluator Results

```json
{
  "evaluator_type": "dimension",
  "llm": "claude",
  "num_completions": 3,
  "dimensions": {
    "relevance": {
      "responses": [
        "First response for relevance...",
        "Second response for relevance...",
        "Third response for relevance..."
      ],
      "weight": 0.25
    }
    // Other dimensions...
  }
}
```

### LLM-based BERTScore Results

```json
{
  "evaluator_type": "bertscore",
  "implementation": "llm-based",
  "llm": "claude",
  "num_completions": 3,
  "raw_responses": [
    "First full response from LLM...",
    "Second full response from LLM...",
    "Third full response from LLM..."
  ],
  "scores": {
    "overall": {
      "precision": 0.85,
      "recall": 0.78,
      "f1": 0.81
    },
    "individual_scores": [
      {"precision": 0.84, "recall": 0.76, "f1": 0.80},
      {"precision": 0.86, "recall": 0.79, "f1": 0.82},
      {"precision": 0.85, "recall": 0.79, "f1": 0.82}
    ]
  }
}
```

## How It Works

1. **Configuration Loading**:
   - Environment variables are loaded
   - Configuration file is parsed
   - Command-line arguments override config settings

2. **File Processing**:
   - File pairs are loaded from explicit configuration or by scanning directories
   - File contents are read and metadata is collected

3. **Evaluation**:
   - If multiple evaluators are configured:
     - Each evaluator is run in sequence
     - Results from all evaluators are aggregated
   - For dimension-based evaluation:
     - For each dimension, the configured LLM generates the specified number of responses
   - For LLM-based BERTScore evaluation:
     - The LLM is prompted to analyze semantic similarity
     - The response is parsed to extract semantic similarity metrics

4. **Results Aggregation**:
   - Results are structured based on the evaluator type(s)
   - Original texts, filenames, timestamps, and metadata are included
   - Results are saved as JSON files in the output directory

