from app.backend.adversarial_suffixes import DUMMY_ADV_SUFFIXES


def test_dummy_suffixes_are_non_empty_list():
    assert isinstance(DUMMY_ADV_SUFFIXES, list)
    assert DUMMY_ADV_SUFFIXES, "expected at least one placeholder suffix"


def test_dummy_suffixes_are_distinct_non_empty_strings():
    assert all(isinstance(s, str) and s.strip() for s in DUMMY_ADV_SUFFIXES)
    assert len(set(DUMMY_ADV_SUFFIXES)) == len(DUMMY_ADV_SUFFIXES)
