from app.backend.adversarial_suffixes import ADV_SUFFIXES


def test_adv_suffixes_are_non_empty_list():
    assert isinstance(ADV_SUFFIXES, list)
    assert ADV_SUFFIXES, "expected at least one adversarial suffix"


def test_adv_suffixes_are_distinct_non_empty_strings():
    assert all(isinstance(s, str) and s.strip() for s in ADV_SUFFIXES)
    assert len(set(ADV_SUFFIXES)) == len(ADV_SUFFIXES)
