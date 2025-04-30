# Text Evaluation Metrics

TMEval supports multiple evaluation metrics, each providing different insights into text quality. This document provides details about the available metrics and how to use them.

## Available Metrics

### 1. Dimension-based (LLM)

Uses Large Language Models to evaluate text quality across multiple dimensions:
- **Relevance**: How well the input text captures important content from the reference
- **Consistency**: Factual alignment between the input text and reference text
- **Fluency**: Quality of individual sentences in the input text
- **Coherence**: How well sentences flow together as a whole

### 2. BERTScore

Uses contextualized embeddings from BERT-like models to measure semantic similarity:
- **Precision**: How much of the input text is semantically present in the reference
- **Recall**: How much of the reference text is covered by the input text
- **F1 Score**: Harmonic mean of precision and recall

### 3. BLEU (Bilingual Evaluation Understudy)

A precision-based metric that counts matching n-grams between the input and reference:
- Calculates matching n-grams (1-grams through 4-grams by default)
- Applies brevity penalty for short translations
- Widely used in machine translation and summarization

### 4. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

A set of metrics that count matching n-grams, focusing on recall:
- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap
- **ROUGE-L**: Longest Common Subsequence (LCS)
- Each variant provides precision, recall, and F-measure scores

## Configuration

Configure which metrics to use in `config.yaml`:

```yaml
evaluator:
  types: 
    - "dimension"  # LLM-based evaluation
    - "bertscore"  # Semantic similarity
    - "bleu"       # N-gram precision
    - "rouge"      # N-gram recall
```

## Metric-Specific Configuration

### BERTScore

```yaml
bertscore:
  implementation: "local-model"  # Uses a local model
  model: "roberta-large"  # Transformer model to use
```

### BLEU

```yaml
bleu:
  max_ngram: 4  # Maximum n-gram order to consider
  weights: [0.25, 0.25, 0.25, 0.25]  # Weights for 1-gram through 4-gram
```

### ROUGE

```yaml
rouge:
  use_stemmer: true  # Whether to apply stemming
  rouge_types: ["rouge1", "rouge2", "rougeL"]  # ROUGE variants to calculate
```

## Running Specific Metrics

Run a specific metric:

```bash
python main.py --evaluator bleu
```

Run multiple metrics:

```bash
python main.py --evaluators dimension,bertscore,bleu,rouge
```

## Understanding Results

### BLEU Results

```json
{
  "evaluator_type": "bleu",
  "max_ngram": 4,
  "weights": [0.25, 0.25, 0.25, 0.25],
  "scores": {
    "overall": 0.651,
    "ngram_scores": {
      "1-gram": 0.825,
      "2-gram": 0.701,
      "3-gram": 0.598,
      "4-gram": 0.480
    },
    "sentence_level": [
      {
        "input": "Climate change involves long-term shifts in temperature and weather patterns.",
        "bleu": 0.732
      },
      // More sentences...
    ]
  }
}
```

### ROUGE Results

```json
{
  "evaluator_type": "rouge",
  "rouge_types": ["rouge1", "rouge2", "rougeL"],
  "use_stemmer": true,
  "scores": {
    "overall": {
      "rouge1": {
        "precision": 0.532,
        "recall": 0.671,
        "fmeasure": 0.593
      },
      "rouge2": {
        "precision": 0.312,
        "recall": 0.394,
        "fmeasure": 0.348
      },
      "rougeL": {
        "precision": 0.487,
        "recall": 0.615,
        "fmeasure": 0.543
      }
    },
    "sentence_level": [
      {
        "input": "Climate change involves long-term shifts in temperature and weather patterns.",
        "scores": {
          "rouge1": {
            "precision": 0.682,
            "recall": 0.450,
            "fmeasure": 0.542
          },
          // More ROUGE scores...
        }
      },
      // More sentences...
    ]
  }
}
```

## Dependencies

- **BLEU**: Requires NLTK
- **ROUGE**: Requires rouge-score package
- **BERTScore**: Requires bert-score, torch, transformers

## Performance Considerations

- BLEU and ROUGE are computationally efficient
- BERTScore is more resource-intensive but offers deeper semantic analysis
- Dimension-based evaluation requires API calls to LLMs and may incur costs
