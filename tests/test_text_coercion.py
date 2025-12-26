import pandas as pd


def test_llm_text_coercion_handles_non_string():
    """Ensure llm_label_fix coerces non-string text (floats/NaN) to str and doesn't crash.

    This test does not require sentence-transformers or network access because
    `llm_label_fix` uses a local mock LLM when none is provided.
    """
    from main import AutoCleanPipeline

    df = pd.DataFrame({
        "id": [1, 2, 3],
        # include NaN, a numeric, and a string that the mock should label 'location'
        "text": [float('nan'), 123.45, "NYC"],
        "label": [None, None, None],
    })

    p = AutoCleanPipeline("in.csv", "out.csv")
    p.df = df

    # Should not raise
    p.llm_label_fix()

    # The mock LLM maps 'NYC' -> 'location'
    assert p.df.loc[2, "label"] == "location"

    # Ensure function didn't crash and at least one label was suggested
    assert p.df["label"].notnull().sum() >= 1
