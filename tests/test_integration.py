import pytest


@pytest.mark.integration
def test_semantic_deduplication_with_real_model():
    """Integration test: runs semantic_deduplication with the real model.

    This test is intentionally guarded by pytest.importorskip so it will be
    skipped when `sentence_transformers` is not installed (fast CI/dev runs).
    To run this locally: `pip install -r requirements-ml.txt`.
    """
    sentence_transformers = pytest.importorskip('sentence_transformers')

    # Build a small DataFrame and run the pipeline end-to-end for dedupe
    import pandas as pd
    from main import AutoCleanPipeline

    df = pd.DataFrame({
        'id': [1, 2],
        'text': ['New York City', 'NYC'],
        'label': ['location', 'location']
    })

    pipeline = AutoCleanPipeline.__new__(AutoCleanPipeline)
    pipeline.df = df.copy()
    pipeline.model = None
    # Run semantic deduplication using the real model (will download if needed)
    pipeline.semantic_deduplication(threshold=0.8)

    # After dedupe, expect one row remaining
    assert len(pipeline.df) == 1