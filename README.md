# AutoClean AI - Data-Centric Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
