from app.backend import constants


def test_hf_fallback_model_is_set():
    assert isinstance(constants.HF_FALLBACK_MODEL, str)
    assert constants.HF_FALLBACK_MODEL  # non-empty
    # All HF model ids are namespace/name.
    assert "/" in constants.HF_FALLBACK_MODEL


def test_hf_token_is_str_or_none():
    assert constants.HF_TOKEN is None or isinstance(constants.HF_TOKEN, str)


def test_hf_cache_dir_is_str_or_none():
    assert constants.HF_CACHE_DIR is None or isinstance(constants.HF_CACHE_DIR, str)
