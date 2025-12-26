<!-- Auto-generated guidance for code-assist agents. Keep concise and actionable. -->
# Copilot / AI Agent Instructions — AutoClean AI

These notes give a concise, actionable summary for AI coding agents to be immediately productive in this repository.

## Big picture (what this repo does)
- A small data-centric pipeline that audits, semantically deduplicates, and exports cleaned CSVs.
- Entry point: `main.py` (class `AutoCleanPipeline`). The pipeline flow is: load -> audit -> semantic_dedup -> llm fix (stub) -> save.

## Key files to read first
- `main.py` — primary implementation. Look at:
  - `AutoCleanPipeline.__init__` (model load: `SentenceTransformer('all-MiniLM-L6-v2')`) — model downloads at init.
  - `load_data()` — `pandas.read_csv` (basic error handling).
  - `semantic_deduplication()` — picks the first object/string column, encodes with the sentence-transformers model, computes cosine similarity, and drops the higher-indexed duplicate (j) when score > 0.85.
  - `llm_label_fix()` — placeholder for LLM-based fixes; integration point for external LLMs (GPT, Ollama, etc.).
- `README.md` — project goals, quick-start, and sample CSVs (`messy_data.csv`, `cleaned_data.csv`).
- `requirements.txt` — packages to install (`sentence-transformers`, `pandas`, `pydantic`, `cleanlab`, etc.).

## Concrete development & debug workflows
- Install and run locally (Python 3.9+):
  - `python3 -m venv venv && source venv/bin/activate`
  - `pip install -r requirements.txt`
  - Run the pipeline: `python main.py --file messy_data.csv --output cleaned_data.csv`
- Logging: `logging.basicConfig(level=logging.INFO)` is used. To get more detail, change `level=logging.DEBUG` at the top of `main.py` or set a runtime env that modifies the logger.
- Model downloads: `SentenceTransformer('all-MiniLM-L6-v2')` downloads weights on first run. Tests or CI must allow network access or mock the model.

## Project-specific conventions & patterns
- Data assumptions: CSV rows have at least one string column used for semantic dedupe. The repository includes an illustrative `DataSchema` Pydantic model (`id: int, text: str, label: Optional[str]`). Use this shape when adding validation.
- Semantic dedupe behavior:
  - The pipeline selects the first object dtype column as the text column.
  - Embeddings are computed for the whole column; pairwise cosine similarity is computed and any pair with score > 0.85 is considered duplicate.
  - When duplicates are found the higher-index row `j` is dropped. This index-based removal is simple but fragile when rows are pre-filtered or shuffled — prefer marking rows and dropping by original index if adding robust dedupe logic.
- LLM integration point: `llm_label_fix()` is explicitly a stub. This is where to add calls to an LLM provider and to wire in `cleanlab` results for focused corrections.

## Integration points & external dependencies
- HuggingFace sentence-transformers (`all-MiniLM-L6-v2`) — used directly in `main.py`.
- Pydantic — `DataSchema` is present but not currently enforced across the pipeline; consider using it in `load_data()` to validate rows.
- Cleanlab — listed in `requirements.txt` but not yet used; `llm_label_fix()` is the expected home for cleanlab-driven label noise detection.

## Quick examples for common agent tasks
- Add a CLI flag to tweak similarity threshold: modify `semantic_deduplication(self, threshold=0.85)` and expose `--threshold` in `argparse` in `__main__`.
- To implement LLM fixes: update `llm_label_fix()` to accept an API client; call the LLM for rows flagged by `cleanlab`/low-confidence heuristics and apply deterministic updates to `self.df`.

## Safety notes for changes
- Avoid changing the default `SentenceTransformer` model name without adding a fallback; CI and dev machines may not have GPU resources.
- When editing dedupe logic, preserve the existing index-removal semantics or explicitly document changes: tests or sample CSVs should demonstrate behavior.

## Where to add tests & what to assert
- Unit tests should exercise:
  - `load_data()` with missing/extra columns
  - `semantic_deduplication()` on a small crafted CSV (ensure duplicates removed as expected)
  - `llm_label_fix()` behavior can be mocked to assert calls but keep the stub until an LLM client is added

## Next small improvements an agent can safely make
- Add a `--threshold` CLI option and unit test for dedupe.
- Use `DataSchema` to validate rows on load and log/skip invalid lines.
- Add a small `tests/test_pipeline.py` that runs pipeline methods on an in-memory DataFrame.

If any of these notes are unclear or you'd like me to expand a specific area (tests, CI, or LLM wiring), tell me which one to implement next.
