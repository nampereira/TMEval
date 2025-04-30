# TMEval Installation Guide

This guide covers the installation of TMEval and its dependencies. TMEval is modular, and different evaluators require different dependencies.

## Basic Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd tmeval
   ```

2. Install core dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up API keys in a `.env` file:
   ```
   CLAUDE_API_KEY=your_claude_api_key
   OPENAI_API_KEY=your_openai_api_key
   GEMINI_API_KEY=your_gemini_api_key
   ```

## Using the Dependency Installer

TMEval includes a helper script to check and install dependencies:

```bash
python install_dependencies.py
```

This script will:
- Check which dependencies are already installed
- Show which evaluators require which dependencies
- Offer to install any missing packages

### Installation Options

Install specific dependencies:

```bash
# Install only ROUGE dependencies
python install_dependencies.py --rouge

# Install only BERTScore dependencies
python install_dependencies.py --bertscore

# Install only BLEU dependencies
python install_dependencies.py --bleu

# Install all dependencies
python install_dependencies.py --all
```

## Required Dependencies by Evaluator

### Core Dependencies (for all evaluators)
- pyyaml
- requests
- python-dotenv
- LLM-specific packages:
  - anthropic (for Claude)
  - openai (for ChatGPT)
  - google-generativeai (for Gemini)

### Dimension-based Evaluator
- No additional dependencies beyond core dependencies

### BERTScore Evaluator
- bert-score
- torch
- transformers
- nltk

### BLEU Evaluator
- nltk

### ROUGE Evaluator
- rouge-score
- nltk

## Manual Installation

If you prefer to install dependencies manually:

### For ROUGE evaluator:
```bash
pip install rouge-score
```

### For BLEU evaluator:
```bash
pip install nltk
python -c "import nltk; nltk.download('punkt')"
```

### For BERTScore evaluator:
```bash
pip install bert-score torch transformers
```

## Troubleshooting

### Missing NLTK Data

If you encounter errors about missing NLTK resources:

```python
import nltk
nltk.download('punkt')
```

### GPU Support for BERTScore

For faster BERTScore evaluation with GPU support:

1. Install PyTorch with CUDA support:
   ```bash
   pip install torch==1.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
   ```
   (Replace with the appropriate version for your CUDA installation)

2. Verify GPU is detected:
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_name(0))
   ```

### Package Conflicts

If you encounter dependency conflicts:
- Consider using a virtual environment: `python -m venv venv`
- Activate the environment before installing dependencies
- On Windows: `venv\Scripts\activate`
- On macOS/Linux: `source venv/bin/activate`

## Running Without Required Dependencies

TMEval is designed to gracefully handle missing dependencies:

- If you run an evaluator without its required dependencies, it will display an error message with installation instructions
- You can still use other evaluators if their dependencies are installed
