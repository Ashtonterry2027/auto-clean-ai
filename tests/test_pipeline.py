import pandas as pd
import numpy as np
import main
from main import AutoCleanPipeline


def test_semantic_deduplication_removes_semantic_duplicates(monkeypatch):
    # Create a DataFrame with two semantically-identical rows and one distinct row
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'text': ['The quick brown fox', 'The quick brown fox', 'A different sentence'],
        'label': [None, None, None]
    })

    # Instantiate pipeline without running __init__ to avoid model download
    pipeline = AutoCleanPipeline.__new__(AutoCleanPipeline)
    pipeline.df = df.copy()

    # Provide a dummy model with an encode method (not actually used by our mocked cos_sim)
    class MockModel:
        def encode(self, sentences, convert_to_tensor=True):
            return None

    pipeline.model = MockModel()

    # Mock util.cos_sim to return a matrix that marks rows 0 and 1 as highly similar
    cosine = np.array([
        [1.0, 0.95, 0.1],
        [0.95, 1.0, 0.05],
        [0.1, 0.05, 1.0]
    ])

    monkeypatch.setattr(main.util, 'cos_sim', lambda a, b: cosine)

    # Run dedupe with threshold high enough to only collapse the first pair
    pipeline.semantic_deduplication(threshold=0.9)

    # Expect one of the duplicate rows removed -> length 2
    assert len(pipeline.df) == 2
    assert 'The quick brown fox' in pipeline.df['text'].values


def test_llm_label_fix_applies_suggestions():
    # Create a DataFrame with a missing label, a correct location label, and an animal label
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'text': ['NYC', 'New York City', 'A fox'],
        'label': [None, 'location', 'animal']
    })

    pipeline = AutoCleanPipeline.__new__(AutoCleanPipeline)
    pipeline.df = df.copy()
    # Ensure the text column is recognized as object dtype
    pipeline.df['text'] = pipeline.df['text'].astype(object)

    class MockLLM:
        def suggest_label(self, text, current_label):
            t = (text or '').lower()
            if 'city' in t or 'nyc' in t:
                return 'location'
            if 'fox' in t:
                return 'animal'
            return current_label

    pipeline.llm_label_fix(llm_client=MockLLM())

    # The missing label should be filled with 'location'
    assert pipeline.df.loc[0, 'label'] == 'location'
    # Existing animal label should remain or be confirmed
    assert pipeline.df.loc[2, 'label'] == 'animal'
