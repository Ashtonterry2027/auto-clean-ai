import pandas as pd
import argparse
import logging
"""AutoClean pipeline.

This module lazily imports heavy ML deps (sentence-transformers) to keep tests
and CI lightweight. A small util shim (`util`) is provided so tests can monkey
patch `main.util.cos_sim` without requiring the real library to be installed.
"""
from pydantic import BaseModel, ValidationError
from typing import Optional
import numpy as np
from collections import Counter
import scipy.stats as stats



class _UtilShim:
    """Lightweight shim that exposes a `cos_sim(a, b)` function and lazily
    delegates to `sentence_transformers.util.cos_sim` when available.
    Tests can monkeypatch `main.util.cos_sim` for deterministic behavior.
    """
    def cos_sim(self, a, b):
        try:
            from sentence_transformers import util as _st_util
            return _st_util.cos_sim(a, b)
        except Exception:
            # Fallback implementation using numpy (works for small arrays/tensors)
            import numpy as _np
            A = _np.array(a)
            B = _np.array(b)
            # Normalize rows
            An = A / (_np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
            Bn = B / (_np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
            return _np.dot(An, Bn.T)


util = _UtilShim()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataSchema(BaseModel):
    # Example schema, strictly for validation demonstration
    id: int
    text: str
    label: Optional[str] = None

class AutoCleanPipeline:
    def __init__(self, file_path: str, output_path: str, llm_client: Optional[object] = None, threshold: float = 0.85):
        self.file_path = file_path
        self.output_path = output_path
        self.df = None
        # Defer heavy model instantiation until semantic_deduplication runs.
        self.model = None
        # Optional injected LLM client. Must implement suggest_label(text, current_label) -> Optional[str]
        self.llm_client = llm_client
        # Default semantic similarity threshold (can be overridden via CLI)
        self.threshold = threshold

    def load_data(self):
        logging.info(f"Loading data from {self.file_path}...")
        try:
            self.df = pd.read_csv(self.file_path)
            logging.info(f"Loaded {len(self.df)} rows.")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def audit_data(self):
        logging.info("--- Step 1: Auditing Data ---")
        # Basic stats
        missing_values = self.df.isnull().sum().sum()
        duplicates = self.df.duplicated().sum()
        
        logging.info(f"Missing values: {missing_values}")
        logging.info(f"Exact duplicates: {duplicates}")
        
        # Calculate a simple health score
        total_cells = self.df.size
        health_score = 100 - ((missing_values + duplicates) / total_cells * 100)
        logging.info(f"Initial Data Health Score: {health_score:.2f}/100")

    def detect_outliers(self, factor=1.5):
        logging.info("--- Step 1.5: Detecting Outliers (IQR Method) ---")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        # Exclude 'id' from outlier detection
        numeric_cols = [c for c in numeric_cols if c.lower() != 'id']
        
        if len(numeric_cols) == 0:
            logging.info("No numeric columns found for outlier detection.")
            return

        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (factor * IQR)
            upper_bound = Q3 + (factor * IQR)
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            if not outliers.empty:
                logging.warning(f"Found {len(outliers)} outliers in column '{col}' (Bounds: {lower_bound:.2f}, {upper_bound:.2f}):")
                for idx, row in outliers.iterrows():
                    logging.info(f"  - Row {idx}: value={row[col]}")
            else:
                logging.info(f"No outliers detected in column '{col}'.")

    def find_label_issues(self):
        logging.info("--- Step 2.5: Finding Label Issues (Cleanlab) ---")
        if 'label' not in self.df.columns or self.df['label'].isnull().any():
            logging.warning("Cleanlab requires a complete 'label' column. Skipping.")
            return

        try:
            import cleanlab
            from sklearn.preprocessing import LabelEncoder
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_predict

            # Ensure we have embeddings
            text_columns = self.df.select_dtypes(include=['object']).columns
            if len(text_columns) == 0:
                return
            
            target_col = text_columns[0]
            sentences = self.df[target_col].fillna("").tolist()
            
            if self.model is None:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            X = self.model.encode(sentences)
            le = LabelEncoder()
            y = le.fit_transform(self.df['label'].astype(str))
            
            if len(np.unique(y)) < 2:
                logging.info("Fewer than 2 classes; skipping label issue detection.")
                return

            # Get out-of-sample predicted probabilities
            clf = LogisticRegression(max_iter=1000)
            
            # Dynamically determine folds based on min class size
            class_counts = Counter(y)
            min_class_size = min(class_counts.values())
            
            if min_class_size < 2:
                logging.warning("Some classes have only 1 member; Cleanlab may be less accurate.")
                num_cross_val_folds = 2 # Minimum possible
            else:
                num_cross_val_folds = min(min_class_size, 5)

            if len(y) < num_cross_val_folds:
                 logging.warning("Not enough data for cross-validation. Skipping Cleanlab.")
                 return

            pred_probs = cross_val_predict(clf, X, y, cv=num_cross_val_folds, method='predict_proba')
            
            # Find issues
            issues = cleanlab.filter.find_label_issues(y, pred_probs)
            if issues.any():
                issue_indices = np.where(issues)[0]
                logging.warning(f"Cleanlab found {len(issue_indices)} potential label issues at indices: {issue_indices}")
                for idx in issue_indices:
                    logging.info(f"Issue at idx {idx}: text='{sentences[idx]}', current_label='{self.df.iloc[idx]['label']}'")
            else:
                logging.info("No label issues detected by Cleanlab.")

        except Exception as e:
            logging.error(f"Error in Cleanlab analysis: {e}")


    def semantic_deduplication(self, threshold=0.85):
        logging.info("--- Step 2: Semantic De-duplication ---")
        # Assuming there is a text column to compare. If not, we'll skip or use the first string column.
        text_columns = self.df.select_dtypes(include=['object']).columns
        if len(text_columns) == 0:
            logging.warning("No text columns found for semantic analysis.")
            return

        target_col = text_columns[0] # simplistic: pick first text column
        logging.info(f"Analyzing semantic similarity on column: '{target_col}'")
        
        sentences = self.df[target_col].fillna("").tolist()
        # Ensure the heavy model is instantiated lazily. This avoids downloading
        # weights during tests or when the ML deps aren't installed.
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                raise RuntimeError("sentence-transformers is required for semantic deduplication; install requirements-ml.txt")

        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        
        cosine_scores = util.cos_sim(embeddings, embeddings)
        
        # Find pairs
        duplicates_found = []
        rows_to_drop = set()
        
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                if cosine_scores[i][j] > threshold:
                    logging.info(f"Found match: '{sentences[i]}' == '{sentences[j]}' (Score: {cosine_scores[i][j]:.4f})")
                    duplicates_found.append((i, j))
                    rows_to_drop.add(j) # Mark the second one for removal
        
        if rows_to_drop:
            logging.info(f"Removing {len(rows_to_drop)} semantic duplicates...")
            self.df = self.df.drop(list(rows_to_drop)).reset_index(drop=True)
        else:
            logging.info("No semantic duplicates found.")

    def llm_label_fix(self, llm_client: Optional[object] = None):
        """LLM-in-the-loop labeling helper.

        Heuristics used to flag candidates:
        - missing labels
        - conflicting labels for identical text (same text -> multiple different labels)
        - rare labels (occurring only once)

        `llm_client` should implement `suggest_label(text, current_label) -> Optional[str]`.
        If None, a lightweight non-networking mock is used.
        """
        logging.info("--- Step 3: LLM-in-the-loop Labeling ---")

        if 'label' not in self.df.columns:
            logging.info("No 'label' column found; skipping LLM label fixes.")
            return

        # 1) Missing labels
        missing_mask = self.df['label'].isnull() | (self.df['label'].astype(str).str.strip() == "")

        # 2) Conflicting labels for identical text (same text, multiple labels)
        text_col = None
        text_columns = self.df.select_dtypes(include=['object']).columns
        if len(text_columns) > 0:
            text_col = text_columns[0]

        conflict_idxs = set()
        if text_col is not None:
            grouped = self.df.groupby(text_col)['label'].nunique()
            conflicts = grouped[grouped > 1].index.tolist()
            if conflicts:
                conflict_rows = self.df[self.df[text_col].isin(conflicts)].index.tolist()
                conflict_idxs.update(conflict_rows)

        # 3) Rare labels (occurrence <= 1)
        label_counts = Counter(self.df['label'].astype(str).tolist())
        rare_labels = {lbl for lbl, cnt in label_counts.items() if cnt <= 1}
        rare_mask = self.df['label'].astype(str).isin(rare_labels)

        candidate_mask = missing_mask | rare_mask
        candidate_idxs = set(self.df[candidate_mask].index.tolist())
        candidate_idxs.update(conflict_idxs)

        if not candidate_idxs:
            logging.info("No candidate rows found for LLM relabeling.")
            return

        logging.info(f"Found {len(candidate_idxs)} candidate rows for LLM review.")

        # Default mock client (non-networking) if none provided
        class _MockLLMClient:
            def suggest_label(self, text: str, current_label: Optional[str]) -> Optional[str]:
                t = (text or "").lower()
                if 'city' in t or 'nyc' in t or 'new york' in t:
                    return 'location'
                if 'fox' in t or 'dog' in t or 'cat' in t or 'animal' in t:
                    return 'animal'
                return current_label

        llm = llm_client or getattr(self, 'llm_client', None) or _MockLLMClient()

        changes = []
        for idx in sorted(candidate_idxs):
            row = self.df.loc[idx]
            # Coerce text to string safely (handle NaN/float/etc.) before sending to LLM
            if text_col is not None:
                raw_text = row[text_col]
                text = "" if pd.isna(raw_text) else str(raw_text)
            else:
                text = ""

            current_label = None if pd.isna(row['label']) else str(row['label'])

            try:
                suggested = llm.suggest_label(text, current_label)
            except Exception as e:
                logging.warning(f"LLM client failed for row {idx}: {e}")
                suggested = current_label

            if suggested is not None and suggested != current_label:
                logging.info(f"Row {idx}: label '{current_label}' -> '{suggested}' (text: '{text}')")
                self.df.at[idx, 'label'] = suggested
                changes.append((idx, current_label, suggested))

        logging.info(f"LLM Agent suggested {len(changes)} corrections.")

    def save_data(self):
        logging.info(f"--- Step 4: Exporting ---")
        self.df.to_csv(self.output_path, index=False)
        logging.info(f"Cleaned data saved to {self.output_path}")

    def run(self):
        self.load_data()
        self.audit_data()
        self.detect_outliers()
        # Use configured threshold and llm client
        self.semantic_deduplication(threshold=self.threshold)
        self.find_label_issues()
        self.llm_label_fix(llm_client=self.llm_client)
        self.save_data()
        logging.info("Pipeline completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoClean AI Data Pipeline")
    parser.add_argument("--file", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to output cleaned CSV file")
    parser.add_argument("--threshold", type=float, default=0.85, help="Semantic similarity threshold (default: 0.85)")
    
    args = parser.parse_args()
    
    try:
        pipeline = AutoCleanPipeline(args.file, args.output, threshold=args.threshold)
        pipeline.run()
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
