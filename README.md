# AutoClean AI - Data-Centric Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/Ashtonterry2027/auto-clean-ai/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Ashtonterry2027/auto-clean-ai/actions/workflows/ci.yml)
[![Codecov](https://codecov.io/gh/Ashtonterry2027/auto-clean-ai/branch/main/graph/badge.svg)](https://codecov.io/gh/Ashtonterry2027/auto-clean-ai)

**An LLM-powered agent for automated data auditing, cleaning, and semantic labeling.**

## 🎯 The Problem

Data scientists spend 80% of their time cleaning messy data. Traditional rule-based cleaning (regex/scripts) fails when data is semantically inconsistent (e.g., "NYC" vs "New York City" vs "The Big Apple") or when labels are noisy.

## 🚀 The Solution

AutoClean AI is a Data-Centric AI pipeline that uses **Small Language Models (SLMs)** and **Sentence Embeddings** to audit datasets, identify anomalies, and programmatically suggest corrections. It goes beyond simple deduplication by understanding the *meaning* of your data.

### Key Features

* **Semantic De-duplication**: Uses HuggingFace Sentence-Transformers to find duplicate records that aren't exact string matches (e.g., "New York" is the same as "NYC").
* **Data Auditing**: Generates a "Data Health Score" based on missing values, duplicates, and other metrics.
* **LLM-in-the-loop Labeling**: (Stubbed) Architecture designed to use LLMs to fix mislabeled training data.
* **Schema Validation**: Ensures strict adherence to data schemas using Pydantic.

## 🛠️ The Tech Stack

* **Core Logic**: Python, Pandas
* **AI Quality**: Cleanlab (for future label noise identification)
* **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
* **Validation**: Pydantic
* **CLI**: Argparse

## 📂 Project Structure

```text
auto-clean-ai/
├── main.py             # Entry point for the pipeline
├── requirements.txt    # Project dependencies
├── messy_data.csv      # Example input file with errors
└── README.md           # Project documentation
```

## ⚡ Quick Start

### 1. Installation

```bash
git clone https://github.com/your-username/auto-clean-ai.git
cd auto-clean-ai

# Create a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Usage

Run the pipeline on your dataset by specifying the input and output paths:

```bash
python main.py --file messy_data.csv --output cleaned_data.csv
```

## 🔬 Integration tests and LLM providers

Integration tests exercise the real ML model and external LLM providers. They are intentionally separated from the fast unit test suite.

Run integration tests locally (slow — may download model weights):

```bash
# optional: create and activate virtualenv
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -r requirements-ml.txt
python -m pytest -q -m integration
```

Providers and credentials
* OpenAI: set `OPENAI_API_KEY` in your environment. The `OpenAIClient` in `llm_client.py` will read this variable by default.
* Ollama: run a local Ollama instance or set `OLLAMA_URL` to your Ollama HTTP endpoint (default `http://localhost:11434`). The `OllamaClient` posts to `{OLLAMA_URL}/api/generate`.

CI notes
* The integration job in `.github/workflows/ci.yml` runs only when manually dispatched and installs `requirements-ml.txt`. If you want CI to call real provider APIs, add repository secrets (e.g., `OPENAI_API_KEY`) in the GitHub repo settings and reference them in the workflow.

HELP: mocking the LLM client for local development
-------------------------------------------------

If you want to develop or test locally without calling external LLM APIs, mock the LLM client:

* Create a small class implementing `suggest_label(text, current_label)` and pass it into the pipeline:

```py
from main import AutoCleanPipeline

class DummyLLM:
  def suggest_label(self, text, current_label):
    # simple deterministic behavior for tests
    t = (text or '').lower()
    if 'nyc' in t or 'new york' in t:
      return 'location'
    if 'fox' in t:
      return 'animal'
    return current_label

pipeline = AutoCleanPipeline('in.csv', 'out.csv', llm_client=DummyLLM())
pipeline.load_data()
pipeline.llm_label_fix()
```

This keeps development fast and offline. The unit tests in `tests/` already demonstrate how to inject a mock LLM client.

Example: GitHub Actions secrets wiring (safe example)
-------------------------------------------------

If you want the integration job to call a real provider (OpenAI) during a manual run, add the secret `OPENAI_API_KEY` to your repository (Settings → Secrets). Then modify the integration job (it already supports manual dispatch) to expose the secret as an environment variable when running tests. In our CI we do this safely by reading the secret only in the integration job.

Example snippet (already wired in `.github/workflows/ci.yml` integration job):

```yaml
   - name: Run integration tests (marked with @pytest.mark.integration)
    env:
     OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    run: |
     python -m pytest -q -m integration
```

When running locally, set the env var temporarily:

```bash
export OPENAI_API_KEY="sk-..."
python -m pytest -q -m integration
```

The code in `llm_client.py` reads `OPENAI_API_KEY` by default if you instantiate `OpenAIClient()` without passing an explicit key.

### 3. Example Result

**Input (`messy_data.csv`)**:
Contains duplicates like "The quick brown fox" and semantic duplicates like "NYC" vs "New York City".

| id | text | label |
|----|------|-------|
| 1 | The quick brown fox | animal |
| 2 | The quick brown fox | animal |
| 3 | A fast brown fox | animal |
| 4 | NYC | location |
| 5 | New York City | location |

**Output (`cleaned_data.csv`)**:
The pipeline identifies that "A fast brown fox" is semantically similar to "The quick brown fox" and that "NYC" matches "New York City", removing the redundancies.

| id | text | label |
|----|------|-------|
| 1 | The quick brown fox | animal |
| 4 | NYC | location |

## 📈 Pipeline Architecture

1. **Audit**: The system scans for outliers and label noise using statistical methods.
2. **Semantic Analysis**: Rows are converted to embeddings (Vector Space) to detect near-duplicates using Cosine Similarity.
3. **Reasoning**: (Planned) An LLM agent reviews "flagged" rows and suggests fixes.
4. **Export**: Cleaned data is exported ready for training.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
