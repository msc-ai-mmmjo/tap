"""Unit tests for ``olmo_tap.final_evals.elo.judge``.

These tests cover the pure logic of the judge pipeline — cache key
derivation, position-swap reconciliation, source filtering, cache I/O,
verdict parsing, and the batch flow with a mocked Anthropic client.
No real API calls are made.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from olmo_tap.final_evals.elo import judge
from olmo_tap.final_evals.elo.judge import (
    JudgeConfig,
    PairToJudge,
    Rubric,
    derive_cache_key,
    filter_pairs_for_dimension,
    judge_pairs,
    load_cache,
    parse_verdict,
    reconcile_swap,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def factuality_rubric(tmp_path: Path) -> Rubric:
    text = "# version: v1\nDIMENSION: factuality\nRUBRIC: judge factuality.\n"
    path = tmp_path / "factuality.txt"
    path.write_text(text, encoding="utf-8")
    return Rubric.load("factuality", path)


@pytest.fixture
def calibration_rubric(tmp_path: Path) -> Rubric:
    text = "# version: v1\nDIMENSION: calibration\nRUBRIC: judge calibration.\n"
    path = tmp_path / "calibration.txt"
    path.write_text(text, encoding="utf-8")
    return Rubric.load("calibration", path)


@pytest.fixture
def basic_pair() -> PairToJudge:
    return PairToJudge(
        prompt_id="srcA_00001",
        source="medmcqa_open",
        prompt_text="What is the first-line treatment for hypothyroidism?",
        entrant_a="entrant_x",
        entrant_b="entrant_y",
        response_a="Levothyroxine, taken daily.",
        response_b="Methimazole.",
        gold_answer="Levothyroxine.",
    )


@pytest.fixture
def curated_pair() -> PairToJudge:
    return PairToJudge(
        prompt_id="srcC_001",
        source="curated",
        prompt_text="Is drug X safe in the first trimester?",
        entrant_a="entrant_x",
        entrant_b="entrant_y",
        response_a="Hedge: insufficient data.",
        response_b="Yes, definitely safe.",
        gold_answer=None,
        expected_behavior="hedge",
    )


# --------------------------------------------------------------------------- #
# Cache key derivation
# --------------------------------------------------------------------------- #


def test_cache_key_is_deterministic(factuality_rubric: Rubric) -> None:
    args = dict(
        rubric_text=factuality_rubric.text,
        judge_model="claude-sonnet-4-6",
        prompt_id="p1",
        entrant_a="a",
        entrant_b="b",
        position_swap=False,
        response_a="hello",
        response_b="world",
    )
    assert derive_cache_key(**args) == derive_cache_key(**args)


def test_cache_key_changes_when_rubric_text_changes(factuality_rubric: Rubric) -> None:
    base = derive_cache_key(
        rubric_text=factuality_rubric.text,
        judge_model="claude-sonnet-4-6",
        prompt_id="p1",
        entrant_a="a",
        entrant_b="b",
        position_swap=False,
        response_a="hello",
        response_b="world",
    )
    perturbed = derive_cache_key(
        rubric_text=factuality_rubric.text + "X",
        judge_model="claude-sonnet-4-6",
        prompt_id="p1",
        entrant_a="a",
        entrant_b="b",
        position_swap=False,
        response_a="hello",
        response_b="world",
    )
    assert base != perturbed


def test_cache_key_changes_with_position_swap(factuality_rubric: Rubric) -> None:
    common = dict(
        rubric_text=factuality_rubric.text,
        judge_model="claude-sonnet-4-6",
        prompt_id="p1",
        entrant_a="a",
        entrant_b="b",
        response_a="hello",
        response_b="world",
    )
    forward = derive_cache_key(position_swap=False, **common)
    swapped = derive_cache_key(position_swap=True, **common)
    assert forward != swapped


def test_cache_key_normalises_line_endings(factuality_rubric: Rubric) -> None:
    args = dict(
        judge_model="claude-sonnet-4-6",
        prompt_id="p1",
        entrant_a="a",
        entrant_b="b",
        position_swap=False,
    )
    crlf = derive_cache_key(
        rubric_text=factuality_rubric.text.replace("\n", "\r\n"),
        response_a="line1\r\nline2",
        response_b="x",
        **args,
    )
    lf = derive_cache_key(
        rubric_text=factuality_rubric.text,
        response_a="line1\nline2",
        response_b="x",
        **args,
    )
    assert crlf == lf


def test_rubric_load_requires_version_header(tmp_path: Path) -> None:
    path = tmp_path / "no_header.txt"
    path.write_text("DIMENSION: factuality\n", encoding="utf-8")
    with pytest.raises(ValueError):
        Rubric.load("factuality", path)


# --------------------------------------------------------------------------- #
# Source filtering
# --------------------------------------------------------------------------- #


def test_factuality_filters_curated(
    basic_pair: PairToJudge, curated_pair: PairToJudge
) -> None:
    filtered = filter_pairs_for_dimension([basic_pair, curated_pair], "factuality")
    assert filtered == [basic_pair]


def test_calibration_keeps_curated(
    basic_pair: PairToJudge, curated_pair: PairToJudge
) -> None:
    filtered = filter_pairs_for_dimension([basic_pair, curated_pair], "calibration")
    assert filtered == [basic_pair, curated_pair]


def test_clinical_utility_keeps_curated(
    basic_pair: PairToJudge, curated_pair: PairToJudge
) -> None:
    filtered = filter_pairs_for_dimension(
        [basic_pair, curated_pair], "clinical_utility"
    )
    assert filtered == [basic_pair, curated_pair]


# --------------------------------------------------------------------------- #
# User-message construction
# --------------------------------------------------------------------------- #


def test_user_message_includes_gold_for_factuality(
    factuality_rubric: Rubric, basic_pair: PairToJudge
) -> None:
    query = judge._JudgeQuery(
        cache_key="dummy",
        pair=basic_pair,
        rubric=factuality_rubric,
        position_swap=False,
        judge_model="claude-sonnet-4-6",
    )
    message = judge.build_user_message(query)
    assert "Gold reference" in message
    assert "Levothyroxine." in message


def test_user_message_omits_gold_for_calibration(
    calibration_rubric: Rubric, basic_pair: PairToJudge
) -> None:
    query = judge._JudgeQuery(
        cache_key="dummy",
        pair=basic_pair,
        rubric=calibration_rubric,
        position_swap=False,
        judge_model="claude-sonnet-4-6",
    )
    message = judge.build_user_message(query)
    assert "Gold reference" not in message


def test_user_message_includes_expected_behavior_for_curated_calibration(
    calibration_rubric: Rubric, curated_pair: PairToJudge
) -> None:
    query = judge._JudgeQuery(
        cache_key="dummy",
        pair=curated_pair,
        rubric=calibration_rubric,
        position_swap=False,
        judge_model="claude-sonnet-4-6",
    )
    message = judge.build_user_message(query)
    assert "expected response pattern is `hedge`" in message


def test_user_message_omits_expected_behavior_for_non_curated_calibration(
    calibration_rubric: Rubric, basic_pair: PairToJudge
) -> None:
    query = judge._JudgeQuery(
        cache_key="dummy",
        pair=basic_pair,
        rubric=calibration_rubric,
        position_swap=False,
        judge_model="claude-sonnet-4-6",
    )
    message = judge.build_user_message(query)
    assert "expected response pattern" not in message


def test_user_message_swaps_responses_when_position_swap_true(
    factuality_rubric: Rubric, basic_pair: PairToJudge
) -> None:
    query = judge._JudgeQuery(
        cache_key="dummy",
        pair=basic_pair,
        rubric=factuality_rubric,
        position_swap=True,
        judge_model="claude-sonnet-4-6",
    )
    message = judge.build_user_message(query)
    a_idx = message.index("Response A:")
    b_idx = message.index("Response B:")
    assert a_idx < b_idx
    # Position-swapped: response_b should now appear in slot A.
    section_a = message[a_idx:b_idx]
    assert basic_pair.response_b in section_a
    assert basic_pair.response_a not in section_a


# --------------------------------------------------------------------------- #
# Verdict parsing
# --------------------------------------------------------------------------- #


def test_parse_verdict_picks_last_line() -> None:
    text = "Some reasoning...\nFinal thought.\nVERDICT: B"
    assert parse_verdict(text) == "B"


def test_parse_verdict_handles_a_b_tie() -> None:
    assert parse_verdict("blah\nVERDICT: A") == "A"
    assert parse_verdict("blah\nVERDICT: B") == "B"
    assert parse_verdict("blah\nVERDICT: TIE") == "TIE"


def test_parse_verdict_falls_back_to_tie_on_garbage() -> None:
    assert parse_verdict("no verdict line here") == "TIE"


def test_parse_verdict_tolerates_trailing_text_on_verdict_line() -> None:
    assert parse_verdict("VERDICT: A wins clearly") == "A"


# --------------------------------------------------------------------------- #
# Position-swap reconciliation
# --------------------------------------------------------------------------- #


def _raw(
    pair: PairToJudge,
    rubric: Rubric,
    *,
    position_swap: bool,
    verdict: judge.Verdict,
) -> judge._RawJudgement:
    return judge._make_raw_for_test(
        pair=pair,
        rubric=rubric,
        judge_model="claude-sonnet-4-6",
        position_swap=position_swap,
        verdict=verdict,
    )


def test_reconcile_consistent_a_wins(
    factuality_rubric: Rubric, basic_pair: PairToJudge
) -> None:
    forward = _raw(basic_pair, factuality_rubric, position_swap=False, verdict="A")
    swapped = _raw(basic_pair, factuality_rubric, position_swap=True, verdict="B")
    judgment = reconcile_swap(forward, swapped, "factuality")
    assert judgment.winner == basic_pair.entrant_a
    assert judgment.inconsistent is False


def test_reconcile_consistent_b_wins(
    factuality_rubric: Rubric, basic_pair: PairToJudge
) -> None:
    forward = _raw(basic_pair, factuality_rubric, position_swap=False, verdict="B")
    swapped = _raw(basic_pair, factuality_rubric, position_swap=True, verdict="A")
    judgment = reconcile_swap(forward, swapped, "factuality")
    assert judgment.winner == basic_pair.entrant_b
    assert judgment.inconsistent is False


def test_reconcile_inconsistent_collapses_to_tie(
    factuality_rubric: Rubric, basic_pair: PairToJudge
) -> None:
    # Both orderings say "the response in slot A wins" — that means each
    # ordering picks a different *entrant*, which is the position-bias
    # failure mode the swap is designed to catch.
    forward = _raw(basic_pair, factuality_rubric, position_swap=False, verdict="A")
    swapped = _raw(basic_pair, factuality_rubric, position_swap=True, verdict="A")
    judgment = reconcile_swap(forward, swapped, "factuality")
    assert judgment.winner is None
    assert judgment.inconsistent is True


def test_reconcile_double_tie_is_consistent(
    factuality_rubric: Rubric, basic_pair: PairToJudge
) -> None:
    forward = _raw(basic_pair, factuality_rubric, position_swap=False, verdict="TIE")
    swapped = _raw(basic_pair, factuality_rubric, position_swap=True, verdict="TIE")
    judgment = reconcile_swap(forward, swapped, "factuality")
    assert judgment.winner is None
    assert judgment.inconsistent is False


def test_reconcile_one_tie_one_winner_is_inconsistent(
    factuality_rubric: Rubric, basic_pair: PairToJudge
) -> None:
    forward = _raw(basic_pair, factuality_rubric, position_swap=False, verdict="A")
    swapped = _raw(basic_pair, factuality_rubric, position_swap=True, verdict="TIE")
    judgment = reconcile_swap(forward, swapped, "factuality")
    assert judgment.winner is None
    assert judgment.inconsistent is True


# --------------------------------------------------------------------------- #
# Cache round-trip
# --------------------------------------------------------------------------- #


def test_cache_round_trip(
    tmp_path: Path,
    factuality_rubric: Rubric,
    basic_pair: PairToJudge,
) -> None:
    cache_dir = tmp_path / "judgments"
    raw = _raw(basic_pair, factuality_rubric, position_swap=False, verdict="A")
    record = judge._raw_to_record(raw, "factuality")
    judge.append_cache_record(cache_dir, "factuality", record)

    reloaded = load_cache(cache_dir, "factuality")
    assert raw.cache_key in reloaded
    reloaded_record = reloaded[raw.cache_key]
    assert reloaded_record["verdict"] == "A"
    assert reloaded_record["judge_model"] == "claude-sonnet-4-6"
    assert reloaded_record["rubric_version"] == "v1"


def test_load_cache_returns_empty_on_missing_file(tmp_path: Path) -> None:
    cache = load_cache(tmp_path / "doesnotexist", "factuality")
    assert cache == {}


def test_load_cache_skips_malformed_lines(tmp_path: Path) -> None:
    cache_dir = tmp_path / "judgments"
    cache_dir.mkdir(parents=True)
    path = cache_dir / "judgments_factuality.jsonl"
    path.write_text(
        '{"cache_key": "abc", "verdict": "A", "judge_model": "m", "rubric_version": "v1"}\n'
        "this is not json\n",
        encoding="utf-8",
    )
    cache = load_cache(cache_dir, "factuality")
    assert list(cache.keys()) == ["abc"]


# --------------------------------------------------------------------------- #
# Batch flow with mocked Anthropic client
# --------------------------------------------------------------------------- #


class _FakeContentBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeUsage:
    def __init__(
        self,
        cache_creation: int = 0,
        cache_read: int = 0,
        input_tokens: int = 100,
        output_tokens: int = 50,
    ) -> None:
        self.cache_creation_input_tokens = cache_creation
        self.cache_read_input_tokens = cache_read
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeMessage:
    def __init__(self, text: str, usage: _FakeUsage) -> None:
        self.content = [_FakeContentBlock(text)]
        self.usage = usage


class _FakeBatchResultEntry:
    def __init__(self, custom_id: str, message: _FakeMessage) -> None:
        self.custom_id = custom_id
        self.result = type("Result", (), {"type": "succeeded", "message": message})()


class _FakeBatch:
    def __init__(self, batch_id: str = "batch_123") -> None:
        self.id = batch_id
        self.processing_status = "ended"


class _FakeBatches:
    """Minimal stand-in for ``client.messages.batches``."""

    def __init__(self, verdicts_by_custom_id: dict[str, str]) -> None:
        self.verdicts_by_custom_id = verdicts_by_custom_id
        self.last_requests: list[dict[str, Any]] | None = None
        self._first_request_seen = False

    def create(self, requests: list[dict[str, Any]]) -> _FakeBatch:
        self.last_requests = requests
        return _FakeBatch()

    def retrieve(self, batch_id: str) -> _FakeBatch:  # noqa: ARG002
        return _FakeBatch(batch_id)

    def results(self, batch_id: str):  # noqa: ARG002
        for custom_id, verdict_text in self.verdicts_by_custom_id.items():
            usage = _FakeUsage(
                cache_creation=200 if not self._first_request_seen else 0,
                cache_read=0 if not self._first_request_seen else 200,
            )
            self._first_request_seen = True
            yield _FakeBatchResultEntry(
                custom_id=custom_id,
                message=_FakeMessage(
                    text=f"Reasoning here.\nVERDICT: {verdict_text}", usage=usage
                ),
            )


class _FakeMessagesNamespace:
    def __init__(self, batches: _FakeBatches) -> None:
        self.batches = batches


class _FakeAnthropic:
    def __init__(self, verdicts_by_custom_id: dict[str, str]) -> None:
        self.messages = _FakeMessagesNamespace(_FakeBatches(verdicts_by_custom_id))


def test_judge_pairs_end_to_end_with_fake_client(
    tmp_path: Path,
    factuality_rubric: Rubric,
    basic_pair: PairToJudge,
) -> None:
    cache_dir = tmp_path / "judgments"
    config = JudgeConfig(
        judge_model="claude-sonnet-4-6",
        cache_dir=cache_dir,
        poll_initial_seconds=0.0,
        poll_max_seconds=0.0,
        api_key="dummy",
    )
    forward_key = derive_cache_key(
        rubric_text=factuality_rubric.text,
        judge_model="claude-sonnet-4-6",
        prompt_id=basic_pair.prompt_id,
        entrant_a=basic_pair.entrant_a,
        entrant_b=basic_pair.entrant_b,
        position_swap=False,
        response_a=basic_pair.response_a,
        response_b=basic_pair.response_b,
    )
    swapped_key = derive_cache_key(
        rubric_text=factuality_rubric.text,
        judge_model="claude-sonnet-4-6",
        prompt_id=basic_pair.prompt_id,
        entrant_a=basic_pair.entrant_a,
        entrant_b=basic_pair.entrant_b,
        position_swap=True,
        response_a=basic_pair.response_a,
        response_b=basic_pair.response_b,
    )
    fake = _FakeAnthropic(
        {
            forward_key: "A",
            swapped_key: "B",
        }
    )

    result = judge_pairs(
        pairs=[basic_pair],
        rubric=factuality_rubric,
        config=config,
        client=fake,
    )

    assert len(result.judgments) == 1
    judgment = result.judgments[0]
    assert judgment.winner == basic_pair.entrant_a
    assert judgment.inconsistent is False
    assert result.cache_stats.fresh_calls == 2
    assert result.cache_stats.cache_creation_input_tokens == 200
    assert result.cache_stats.cache_read_input_tokens == 200

    # Cache file should now contain both entries.
    cache_path = cache_dir / "judgments_factuality.jsonl"
    lines = [
        line for line in cache_path.read_text(encoding="utf-8").splitlines() if line
    ]
    assert len(lines) == 2
    keys = {json.loads(line)["cache_key"] for line in lines}
    assert keys == {forward_key, swapped_key}


def test_judge_pairs_uses_cache_on_second_run(
    tmp_path: Path,
    factuality_rubric: Rubric,
    basic_pair: PairToJudge,
) -> None:
    cache_dir = tmp_path / "judgments"
    config = JudgeConfig(
        judge_model="claude-sonnet-4-6",
        cache_dir=cache_dir,
        poll_initial_seconds=0.0,
        poll_max_seconds=0.0,
        api_key="dummy",
    )
    forward_key = derive_cache_key(
        rubric_text=factuality_rubric.text,
        judge_model="claude-sonnet-4-6",
        prompt_id=basic_pair.prompt_id,
        entrant_a=basic_pair.entrant_a,
        entrant_b=basic_pair.entrant_b,
        position_swap=False,
        response_a=basic_pair.response_a,
        response_b=basic_pair.response_b,
    )
    swapped_key = derive_cache_key(
        rubric_text=factuality_rubric.text,
        judge_model="claude-sonnet-4-6",
        prompt_id=basic_pair.prompt_id,
        entrant_a=basic_pair.entrant_a,
        entrant_b=basic_pair.entrant_b,
        position_swap=True,
        response_a=basic_pair.response_a,
        response_b=basic_pair.response_b,
    )
    fake_first = _FakeAnthropic({forward_key: "A", swapped_key: "B"})
    judge_pairs(
        pairs=[basic_pair],
        rubric=factuality_rubric,
        config=config,
        client=fake_first,
    )

    # On a second run the cache should be hit and the fake client should
    # never have to handle requests.
    fake_second = _FakeAnthropic({})
    second = judge_pairs(
        pairs=[basic_pair],
        rubric=factuality_rubric,
        config=config,
        client=fake_second,
    )
    assert second.cache_stats.fresh_calls == 0
    assert second.cache_stats.cache_hits == 2
    assert fake_second.messages.batches.last_requests is None
    assert second.judgments[0].winner == basic_pair.entrant_a


def test_judge_pairs_skips_curated_for_factuality(
    tmp_path: Path,
    factuality_rubric: Rubric,
    basic_pair: PairToJudge,
    curated_pair: PairToJudge,
) -> None:
    cache_dir = tmp_path / "judgments"
    config = JudgeConfig(
        judge_model="claude-sonnet-4-6",
        cache_dir=cache_dir,
        poll_initial_seconds=0.0,
        poll_max_seconds=0.0,
        api_key="dummy",
    )
    forward_key = derive_cache_key(
        rubric_text=factuality_rubric.text,
        judge_model="claude-sonnet-4-6",
        prompt_id=basic_pair.prompt_id,
        entrant_a=basic_pair.entrant_a,
        entrant_b=basic_pair.entrant_b,
        position_swap=False,
        response_a=basic_pair.response_a,
        response_b=basic_pair.response_b,
    )
    swapped_key = derive_cache_key(
        rubric_text=factuality_rubric.text,
        judge_model="claude-sonnet-4-6",
        prompt_id=basic_pair.prompt_id,
        entrant_a=basic_pair.entrant_a,
        entrant_b=basic_pair.entrant_b,
        position_swap=True,
        response_a=basic_pair.response_a,
        response_b=basic_pair.response_b,
    )
    fake = _FakeAnthropic({forward_key: "A", swapped_key: "B"})

    result = judge_pairs(
        pairs=[basic_pair, curated_pair],
        rubric=factuality_rubric,
        config=config,
        client=fake,
    )
    assert len(result.judgments) == 1
    assert result.judgments[0].prompt_id == basic_pair.prompt_id


def test_judge_pairs_missing_api_key_raises(
    tmp_path: Path,
    factuality_rubric: Rubric,
    basic_pair: PairToJudge,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config = JudgeConfig(
        judge_model="claude-sonnet-4-6",
        cache_dir=tmp_path / "judgments",
        poll_initial_seconds=0.0,
        poll_max_seconds=0.0,
        api_key=None,
    )
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        judge_pairs(pairs=[basic_pair], rubric=factuality_rubric, config=config)


def test_batch_request_carries_cache_control_breakpoint(
    tmp_path: Path,
    factuality_rubric: Rubric,
    basic_pair: PairToJudge,
) -> None:
    cache_dir = tmp_path / "judgments"
    config = JudgeConfig(
        judge_model="claude-sonnet-4-6",
        cache_dir=cache_dir,
        poll_initial_seconds=0.0,
        poll_max_seconds=0.0,
        api_key="dummy",
    )
    forward_key = derive_cache_key(
        rubric_text=factuality_rubric.text,
        judge_model="claude-sonnet-4-6",
        prompt_id=basic_pair.prompt_id,
        entrant_a=basic_pair.entrant_a,
        entrant_b=basic_pair.entrant_b,
        position_swap=False,
        response_a=basic_pair.response_a,
        response_b=basic_pair.response_b,
    )
    swapped_key = derive_cache_key(
        rubric_text=factuality_rubric.text,
        judge_model="claude-sonnet-4-6",
        prompt_id=basic_pair.prompt_id,
        entrant_a=basic_pair.entrant_a,
        entrant_b=basic_pair.entrant_b,
        position_swap=True,
        response_a=basic_pair.response_a,
        response_b=basic_pair.response_b,
    )
    fake = _FakeAnthropic({forward_key: "A", swapped_key: "B"})
    judge_pairs(
        pairs=[basic_pair],
        rubric=factuality_rubric,
        config=config,
        client=fake,
    )

    requests = fake.messages.batches.last_requests
    assert requests is not None and len(requests) == 2
    for request in requests:
        system_block = request["params"]["system"][0]
        assert system_block["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
        # Cache breakpoint is at the *end* of the rubric prefix; per-pair
        # content lives in the user message and must not appear in system.
        assert basic_pair.response_a not in system_block["text"]
        assert basic_pair.response_b not in system_block["text"]
