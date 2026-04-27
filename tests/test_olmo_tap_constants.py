from pathlib import Path

from olmo_tap import constants


def test_vocab_size_is_positive_int():
    assert isinstance(constants.VOCAB_SIZE, int)
    assert constants.VOCAB_SIZE > 0


def test_mcq_letters_are_distinct_single_chars():
    letters = constants.MCQ_LETTERS
    assert len(letters) >= 2
    assert all(isinstance(letter, str) and len(letter) == 1 for letter in letters)
    assert len(set(letters)) == len(letters)


def test_token_limits_are_positive_and_ordered():
    assert constants.MAX_NEW_TOKENS > 0
    assert constants.MCQ_MAX_NEW_TOKENS > 0
    assert constants.DEMO_MAX_NEW_TOKENS > 0
    # MCQ answers fit in fewer tokens than full NLP answers.
    assert constants.MCQ_MAX_NEW_TOKENS <= constants.MAX_NEW_TOKENS


def test_kv_cache_at_least_attack_window():
    assert constants.KV_CACHE_MAX_SEQ_LEN >= constants.ATTACK_MAX_SEQ_LEN > 0


def test_kle_constants_sane():
    assert isinstance(constants.KLE_N_SAMPLES, int)
    assert constants.KLE_N_SAMPLES >= 2
    assert constants.KLE_HEAT_KERNEL_T > 0


def test_lora_targets_non_empty():
    assert constants.LORA_TARGETS
    assert all(isinstance(t, str) and t for t in constants.LORA_TARGETS)
    assert constants.LORA_ALPHA_RATIO > 0


def test_weight_dirs_are_paths():
    assert isinstance(constants.GCG_CACHE_DIR, Path)
    assert isinstance(constants.PROD_WEIGHTS_DIR, Path)
    assert isinstance(constants.ROBUST_WEIGHTS_DIR, Path)
    assert isinstance(constants.UNCERTAINTY_WEIGHTS_DIR, Path)
    assert isinstance(constants.ATTACK_BANK_DIR, Path)
