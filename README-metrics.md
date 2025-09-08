# Text Evaluation Metrics

TMEval supports multiple evaluation metrics, each providing different insights into text quality. This document provides details about the available metrics and how to use them.

## Available Metrics

### 1. Dimension-based (LLM)

Uses Large Language Models to evaluate text quality across multiple dimensions:
- **Plausibility**: Do the listed threats technically align with the inferred system characteristics from the reference?
- **Consistency**: Do the listed threats align with the reference, without unsupported or contradictory information?
- **Coverage**: Vary from STRIDE category to STRIDE category. In general, each one includes the following three dimensions:
  - **Targets Coverage**: Are the same targets identified as affected?
  - **Weaknesses Coverage**: Are similar vulnerabilities or exposures highlighted?
  - **Attacks Coverage**: Are comparable attack methods or strategies described?

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

### Dimension-based

```yaml
dimensions:
  # Coverage dimension per category
  # Denial of Service
  attack-types-coverage:
    prompt_file: "prompts/dos-coverage/attack-types-coverage.txt"
    category: Denial of Service
    weight: 0.20
  dos-protection-gaps-coverage:
    prompt_file: "prompts/dos-coverage/protection-gaps-coverage.txt"
    category: Denial of Service
    weight: 0.20
  resource-coverage:
    prompt_file: "prompts/dos-coverage/resource-coverage.txt"
    category: Denial of Service
    weight: 0.20
  # ... for each category

  # Common dimensions
  consistency:
    prompt_file: "prompts/consistency.txt"
    category: All
    weight: 0.30

  plausibility:
    prompt_file: "prompts/plausibility.txt"
    category: All
    weight: 0.10
```

### BERTScore

```yaml
bertscore:
  implementation: "local-model"  # Uses a local model for BERTScore
  model: "roberta-large"  # Model to use for BERTScore (roberta-large is the default)
  idf: true  # Use inverse document frequency weighting
  rescale_with_baseline: true  # Scale scores to be more interpretable
  verbose: false  # Set to true to see detailed progress and warnings
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
With this command, the corresponding results will be generated in the results directory, which should be reviewed. In addition, a simple interactive dashboard for the respective metric will be launched on port 8050.

Run multiple metrics:

```bash
python main.py --evaluators dimension,bertscore, bleu,rouge
```

## Understanding Results

### Dimension-based Results

```json
{
  "evaluator_type": "dimension",
  "llm": "gemini-1.5-pro",
  "num_completions": 8,
  "dimensions": {
    "asset-coverage": {
      "responses": [
        "Asset Coverage: 4\n",
        "Asset Coverage: 4\n",
        "Asset Coverage: 4\n",
        "Asset Coverage: 4\n",
        "Asset Coverage: 4\n",
        "Asset Coverage: 4\n",
        "Asset Coverage: 4\n",
        "Asset Coverage: 4\n"
      ],
      "weight": 0.2,
      "average": 4.0
    },
    // ... for each dimension
    "score": 3.2875000000000005
  }
}
```

### BERTScore Results

```json
{
  "evaluator_type": "bertscore",
  "implementation": "local-model",
  "model": "roberta-large",
  "idf": true,
  "rescale_with_baseline": true,
  "scores": {
    "overall": {
      "f1": 0.09459596375624339
    },
    "sentence_level": [
      {
        "input": "Since all projects share a single MySQL database server, a SQL injection vulnerability in the backend of one SUNDIAL project could be exploited to modify or delete data in a different project's database (e.g., tampering with the RELiEF project's data from a less secure project's endpoint).  Corruption or destruction of critical and confidential sustainability data for multiple projects, leading to incorrect environmental assessments and invalid research outcomes.",
        "best_reference": "Docker containers if running with root privileges pose a significant security risk. If a container from any project accepts data through API requests without properly validating for malicious shell commands, an attacker could potentially exploit the container.  Could potentially escape the container and compromise other project containers or even the underlying host system.",
        "precision": 0.03307128697633743,
        "recall": 0.14115795493125916,
        "f1": 0.08801961690187454
      },
      // ... for each input
    ]
  }
}
```

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
        "input": "The platform performs Life Cycle Assessments (LCA) and creates complex diagrams. A malicious authenticated user could repeatedly trigger computationally expensive assessments or create models with an excessive number of processes and flows, overwhelming the backend Node.js container's CPU/memory resources.  The application becomes slow or unresponsive for all other users, preventing them from collecting or analyzing data.",
        "best_reference": "An attacker or malicious user could exploit the fact that the MFA process does not have an appropriate consumption limit setting, allowing multiple logins and excessive email links to be sent.  Delays or failures in delivering MFA links for other authentication attempts may occur.",
        "bleu_for_this_sentence": 0.011308221067292446
      },
      // ... for each input
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
        {
          "input": "A malicious user associated with an LCA model could manipulate API requests to edit inventory flow attributes that were created by another user. This bypasses the business rule that 'Only the user who created an entry flow is allowed to edit its attributes' if backend authorization is not properly implemented.  Unauthorized modification of scientific data, compromising the integrity and validity of the entire Life Cycle Assessment.",
          "best_reference": {
            "rouge1": "Docker containers if running with root privileges pose a significant security risk. If a container from any project accepts data through API requests without properly validating for malicious shell commands, an attacker could potentially exploit the container.  Could potentially escape the container and compromise other project containers or even the underlying host system.",
            "rouge2": "Docker containers if running with root privileges pose a significant security risk. If a container from any project accepts data through API requests without properly validating for malicious shell commands, an attacker could potentially exploit the container.  Could potentially escape the container and compromise other project containers or even the underlying host system.",
            "rougeL": "Docker containers if running with root privileges pose a significant security risk. If a container from any project accepts data through API requests without properly validating for malicious shell commands, an attacker could potentially exploit the container.  Could potentially escape the container and compromise other project containers or even the underlying host system."
          },
          "rouge_for_this_sentence": {
            "rouge1": {
              "fmeasure": 0.20253164556962025,
              "precision": 0.19047619047619047,
              "recall": 0.21621621621621623
            },
            "rouge2": {
              "fmeasure": 0.025974025974025976,
              "precision": 0.024390243902439025,
              "recall": 0.027777777777777776
            },
            "rougeL": {
              "fmeasure": 0.10126582278481013,
              "precision": 0.09523809523809523,
              "recall": 0.10810810810810811
            }
          }
        },
      },
      // ... for each input
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
