from app.backend.response_payloads import (
    fallback_robustness,
    fallback_security,
    fallback_uncertainty,
    poe_security,
    poe_uncertainty,
)


def test_fallback_security_shape():
    assert fallback_security() == {
        "certified": None,
        "tokens": [],
        "resampled": [],
        "token_entropies": [],
    }


def test_poe_security_echoes_inputs_and_certifies():
    tokens = ["hello", " world"]
    resampled = [{"index": 1, "alternatives": ["foo", "bar"]}]
    token_entropies = [0.1, 0.4]
    assert poe_security(tokens, resampled, token_entropies) == {
        "certified": True,
        "tokens": tokens,
        "resampled": resampled,
        "token_entropies": token_entropies,
    }


def test_poe_security_with_empty_lists():
    assert poe_security([], [], []) == {
        "certified": True,
        "tokens": [],
        "resampled": [],
        "token_entropies": [],
    }


def test_fallback_uncertainty_shape():
    assert fallback_uncertainty() == {"overall": None}


def test_poe_uncertainty_with_value():
    assert poe_uncertainty(0.73) == {"overall": 0.73}


def test_poe_uncertainty_passes_none_through():
    assert poe_uncertainty(None) == {"overall": None}


def test_fallback_robustness_shape():
    assert fallback_robustness() == {"type": "unavailable"}


def test_poe_uncertainty_with_zero_and_one():
    assert poe_uncertainty(0.0) == {"overall": 0.0}
    assert poe_uncertainty(1.0) == {"overall": 1.0}


def test_fallback_helpers_return_independent_dicts():
    # Mutating one return value must not bleed into the next call. (Cheap
    # safety net against someone refactoring these to module-level constants.)
    a = fallback_security()
    a["tokens"].append("dirty")
    assert fallback_security() == {
        "certified": None,
        "tokens": [],
        "resampled": [],
        "token_entropies": [],
    }

    b = fallback_uncertainty()
    b["overall"] = 1.0
    assert fallback_uncertainty() == {"overall": None}

    c = fallback_robustness()
    c["type"] = "mutated"
    assert fallback_robustness() == {"type": "unavailable"}
