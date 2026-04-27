import pytest

pytest.importorskip("torch")

from kernel_entropy import nli


def test_label_indices_are_distinct_and_cover_three_classes():
    labels = {nli.LABEL_ENTAILMENT, nli.LABEL_NEUTRAL, nli.LABEL_CONTRADICTION}
    assert labels == {0, 1, 2}


def test_default_model_id_is_modernbert_nli():
    assert isinstance(nli.DEFAULT_MODEL_ID, str)
    assert "ModernBERT" in nli.DEFAULT_MODEL_ID
    assert "/" in nli.DEFAULT_MODEL_ID  # namespace/repo
