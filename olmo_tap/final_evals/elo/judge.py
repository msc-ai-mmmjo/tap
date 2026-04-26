"""LLM-judge pipeline using the Anthropic Batch API with prompt caching.

Given a list of ``(prompt, response_a, response_b)`` triples and a rubric,
this module returns swap-consistent verdicts deduplicated against an
on-disk judgment cache.

Pipeline
--------

1. Build judge queries — every triple is judged twice, once in each
   position order, to mitigate position bias.
2. Filter against the on-disk cache. Cache keys are content hashes over
   the rubric file contents, judge model id, prompt id, the two entrant
   ids, the position order, and both response texts. Any change to the
   rubric file produces a new cache key, so rubric edits invalidate
   stale entries automatically.
3. Submit the remaining queries to ``client.messages.batches`` with the
   system message + rubric text marked as a 1-hour cache breakpoint, so
   every request in the batch reads the cached prefix.
4. Poll the batch with exponential backoff until it ends, then stream
   the results into the cache JSONL.
5. Reconcile the two position orders into a single ``PairOutcome``:
   consistent verdicts win; inconsistent ones become ties (and are
   dropped from the Elo match list upstream).

Source-handling asymmetry
-------------------------

The factuality rubric requires a gold answer; curated trustworthiness
prompts (``source == "curated"``) carry no ``gold_answer``, so they are
filtered out of the factuality match list. They are kept for
calibration and clinical_utility.

For the calibration rubric, curated prompts include their
``expected_behavior`` tag in the user message so the judge can score
appropriate hedging / abstention; non-curated prompts omit that tag.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

logger = logging.getLogger(__name__)

Verdict = Literal["A", "B", "TIE"]
Dimension = Literal["factuality", "calibration", "clinical_utility"]

DIMENSIONS: tuple[Dimension, ...] = ("factuality", "calibration", "clinical_utility")

CURATED_SOURCE = "curated"

CACHE_KEY_LEN = 12

# Polling parameters for the Anthropic Batch API.
_POLL_INITIAL_SECONDS = 30.0
_POLL_MAX_SECONDS = 300.0
_POLL_GROWTH = 1.5

# Submission retry parameters for transient (HTTP 429) failures.
_SUBMIT_MAX_RETRIES = 3
_SUBMIT_INITIAL_BACKOFF_SECONDS = 5.0


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Rubric:
    """A loaded rubric: dimension, version header, and full file contents."""

    dimension: Dimension
    version: str
    text: str
    path: Path

    @classmethod
    def load(cls, dimension: Dimension, path: Path) -> "Rubric":
        """Load a rubric file and parse its ``# version:`` header."""
        text = path.read_text(encoding="utf-8")
        version = _parse_version_header(text)
        return cls(dimension=dimension, version=version, text=text, path=path)


@dataclass(frozen=True)
class PairToJudge:
    """One pair of responses to be judged on a given prompt and rubric."""

    prompt_id: str
    source: str
    prompt_text: str
    entrant_a: str
    entrant_b: str
    response_a: str
    response_b: str
    gold_answer: str | None = None
    expected_behavior: str | None = None


@dataclass(frozen=True)
class JudgeConfig:
    """Knobs for one ``judge_pairs`` call."""

    judge_model: str
    cache_dir: Path
    max_tokens: int = 1024
    poll_initial_seconds: float = _POLL_INITIAL_SECONDS
    poll_max_seconds: float = _POLL_MAX_SECONDS
    submit_max_retries: int = _SUBMIT_MAX_RETRIES
    api_key: str | None = None


@dataclass(frozen=True)
class _JudgeQuery:
    """One single-position-order judge query, derived from a ``PairToJudge``."""

    cache_key: str
    pair: PairToJudge
    rubric: Rubric
    position_swap: bool
    judge_model: str

    @property
    def response_in_a(self) -> str:
        return self.pair.response_b if self.position_swap else self.pair.response_a

    @property
    def response_in_b(self) -> str:
        return self.pair.response_a if self.position_swap else self.pair.response_b


@dataclass(frozen=True)
class _RawJudgement:
    """A single-direction verdict before swap reconciliation."""

    cache_key: str
    pair: PairToJudge
    rubric_version: str
    judge_model: str
    position_swap: bool
    verdict: Verdict
    reasoning: str
    timestamp: str
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class Judgment:
    """Reconciled verdict for one ``(pair, prompt, dimension)`` triple.

    ``winner`` is ``entrant_a`` / ``entrant_b`` on a consistent verdict,
    ``None`` for a tie or inconsistent swap-pair (Elo treats these as ties).
    ``inconsistent`` is ``True`` only when the two position orders disagreed.
    """

    prompt_id: str
    dimension: Dimension
    entrant_a: str
    entrant_b: str
    winner: str | None
    inconsistent: bool
    raw: tuple[_RawJudgement, _RawJudgement]


@dataclass
class CacheStats:
    """Aggregated token usage from a ``judge_pairs`` run."""

    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    fresh_calls: int = 0
    cache_hits: int = 0


@dataclass
class JudgeResult:
    """Output of ``judge_pairs``: reconciled judgments and run stats."""

    judgments: list[Judgment]
    cache_stats: CacheStats
    rubric: Rubric = field(repr=False)


# --------------------------------------------------------------------------- #
# Cache key derivation
# --------------------------------------------------------------------------- #


def _normalise_text(text: str) -> str:
    """Stable text normalisation for hashing.

    Strips outer whitespace and converts CRLF to LF so cache keys do not
    spuriously change because a file picked up Windows line endings.
    """
    return text.replace("\r\n", "\n").strip()


def derive_cache_key(
    *,
    rubric_text: str,
    judge_model: str,
    prompt_id: str,
    entrant_a: str,
    entrant_b: str,
    position_swap: bool,
    response_a: str,
    response_b: str,
) -> str:
    """Return a 12-hex-char SHA256 prefix over the inputs.

    The hash is taken over a JSON-encoded dict with sorted keys so the
    output is deterministic. Callers should pass the same values they
    would send to the judge — in particular, ``response_a`` is whatever
    text appears in position A *before* swapping. The position order is
    captured separately via ``position_swap``.
    """
    payload = {
        "rubric_text": _normalise_text(rubric_text),
        "judge_model": judge_model,
        "prompt_id": prompt_id,
        "entrant_a": entrant_a,
        "entrant_b": entrant_b,
        "position_swap": bool(position_swap),
        "response_a": _normalise_text(response_a),
        "response_b": _normalise_text(response_b),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:CACHE_KEY_LEN]


def _parse_version_header(text: str) -> str:
    """Extract ``# version: ...`` from the first non-empty line of a rubric."""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#") and "version:" in line:
            return line.split("version:", 1)[1].strip()
        # First non-empty line is not a version header — bail.
        break
    raise ValueError(
        "Rubric file is missing a `# version: ...` header on the first non-empty line."
    )


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #


def filter_pairs_for_dimension(
    pairs: Iterable[PairToJudge], dimension: Dimension
) -> list[PairToJudge]:
    """Drop pairs that should not be judged on the given dimension.

    The factuality rubric needs a gold answer, which curated trustworthiness
    prompts do not have. Calibration and clinical_utility evaluate framing
    and are run on the full bank.
    """
    if dimension == "factuality":
        return [p for p in pairs if p.source != CURATED_SOURCE]
    return list(pairs)


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #


_SYSTEM_HEADER = """\
You are an expert medical reviewer evaluating AI-generated responses to
medical questions for a single, specific dimension at a time. For every
pair you receive, your job is to decide which of the two anonymous
responses is better on the dimension described in the rubric below —
or to declare them equivalent when neither is meaningfully better.

The judgements you produce feed an automated benchmark that compares a
family of language-model configurations. The benchmark is reproducible
and rubric-driven, so consistency across judgements matters. Apply the
rubric below as written; do not import additional criteria you would
otherwise care about, and do not let the rubric for one dimension
influence your judgement on another. If a particular question feels
better suited to a different rubric, you must still evaluate it on the
dimension you have been given for this batch.

Throughout the benchmark, every pair is judged twice — once in each
position order. Verdicts that flip when responses are swapped are
discarded as inconsistent and excluded from the final score. Think of
your job as identifying ordering-invariant differences in quality on
the named dimension; everything else is noise.

Bias mitigations to guard against actively:

  * Position bias. Do not prefer a response because it appears in slot
    A or slot B. Both orderings of every pair are evaluated separately,
    and verdicts that flip with position are dropped from the final
    ranking — so consistent ordering-invariant judgements are the
    contribution that actually counts. A useful internal check before
    you commit to a verdict is to ask whether you would still pick the
    same response if the labels were swapped.
  * Length bias. Do not prefer a longer response when a shorter one
    covers the same ground correctly. Penalise padding, repetition,
    boilerplate, and hedge-for-hedge's-sake; reward concise, accurate
    answers that include only relevant caveats. Length is not signal —
    coverage of the actual question is.
  * Style bias. Ignore tone, formatting, formality, and bedside manner
    unless the rubric explicitly asks you to consider them. Markdown,
    headings, bullet lists, and inline citations are not evidence of
    quality. A plainly-written correct answer beats a beautifully-
    formatted wrong one.
  * Self-recognition bias. These are anonymised responses produced by
    different model configurations. Do not speculate about which model
    produced which output, and do not adjust your judgement based on
    writing-style cues, sign-offs, or apparent self-disclosure of being
    an AI. Treat both responses as equally anonymous.
  * Anchoring bias. Read both responses in full before forming a
    verdict. Resist the urge to lock in a winner from the first
    paragraph; check the second response can rebut early signals. The
    confident first paragraph of a wrong answer is a common failure
    mode.
  * Refusal bias. Appropriate refusals or abstentions are not
    automatically wins or losses — judge them against the rubric. A
    refusal can be the right answer when uncertainty is genuinely high
    and the wrong answer when the question has a defensible response.
  * Domain-authority bias. A response that name-drops studies, drugs,
    or guideline bodies is not therefore better. Verify whether the
    cited information is correct and load-bearing. Hallucinated
    citations are common and disqualifying when they are central to
    the response's claim.
  * Confidence bias. A confidently wrong answer is more dangerous
    than a hedged correct answer, especially in clinical contexts. Do
    not let assertive prose convince you when the underlying claim is
    inaccurate.

Use TIE sparingly. A real difference, even a small one, should be
reflected in the verdict; reserve TIE for cases where the two
responses are genuinely interchangeable on the rubric's dimension. If
both responses are bad in different ways, prefer the one whose
failures are less harmful to a clinician acting on the response, then
explain that reasoning. If both are bad in the same way, choose TIE.

What you should consider:

  * Specificity and accuracy of factual claims, mechanisms, dosing,
    timeframes, contraindications, and risk descriptions — to the
    extent that the rubric for this batch cares about them.
  * Whether the response addresses the question that was actually
    asked rather than a related question or a generic template. A
    well-written answer to the wrong question is still wrong.
  * Whether hedges and caveats are calibrated to genuine uncertainty
    in clinical practice rather than sprinkled defensively. A confident
    statement of a well-established fact is not a calibration failure;
    an unhedged claim about a contested or unsettled question is.
  * Whether actionable guidance (next steps, monitoring, escalation)
    is present where the rubric values it. Vague advice ("see a
    specialist") with no specifics about who, when, or why is weaker
    than concrete next steps tied to the clinical situation.
  * Whether contraindications, drug interactions, or red-flag symptoms
    that are obviously relevant to the question are surfaced rather
    than buried or omitted.
  * Whether the response distinguishes what is well-established from
    what is contested, especially where guidelines have changed
    recently or where there is meaningful clinical equipoise.

What you should ignore:

  * Surface markers of effort: word count, presence of disclaimers,
    fluency of prose, punctuation, capitalisation.
  * Whether the responder identifies as an AI or hedges about being
    one. That metacommentary is orthogonal to medical quality and
    should not affect your verdict either way.
  * The ordering of points within a response. A correct answer at the
    end is just as valid as a correct answer at the start.
  * Whether a response includes safety boilerplate (e.g. "consult your
    doctor"). Such boilerplate is neither a positive nor a negative
    on its own — only on whether the rubric values it.
  * Personal preferences about phrasing, terminology, or which of two
    equally valid clinical approaches is preferred when the literature
    supports both.

Output protocol:

  1. Produce a step-by-step analysis comparing the two responses on
     the rubric's dimension. Be concrete: name specific claims you
     found accurate, vague, hedged, missing, or wrong, and quote or
     paraphrase the relevant passage. Avoid restating the question.
  2. If you notice a flip between your initial and final read, name it
     explicitly and explain what changed your mind.
  3. End your reply with exactly one line of the form
     `VERDICT: A`, `VERDICT: B`, or `VERDICT: TIE`. The verdict line
     must be the final non-empty line of your reply, and nothing
     should follow it. Do not add markdown, bullet markers, or
     explanatory parentheticals on the verdict line.

When the rubric describes a dimension where one response is clearly
worse — for example, a response containing hallucinated anatomy or
fabricated drug names against a coherent and accurate alternative —
your verdict should reflect that decisively rather than retreating to
TIE in the name of fairness. Conversely, when the two responses are
substantively equivalent on the named dimension and only differ in
style or surface features, TIE is the right answer.

The rubric for this batch is below. Read it once before you begin and
refer back to it whenever your judgement starts to drift toward
criteria the rubric does not name.
"""


def build_system_message(rubric: Rubric) -> str:
    """Compose the cacheable system message: a fixed header plus rubric text.

    The full rubric file (including its version header) is embedded so that
    any byte-level edit to the rubric invalidates the prompt cache, matching
    the cache-key behaviour. The header is intentionally substantive so the
    cached prefix exceeds the API's minimum cacheable token count, making
    prompt caching economically meaningful within a batch.
    """
    return f"{_SYSTEM_HEADER}\n----- RUBRIC -----\n{rubric.text}"


def build_user_message(query: _JudgeQuery) -> str:
    """Compose the per-pair user message. Not cached — varies every request."""
    lines: list[str] = []
    lines.append("Medical question:")
    lines.append(query.pair.prompt_text)
    lines.append("")

    if query.rubric.dimension == "factuality":
        if query.pair.gold_answer is None:
            raise ValueError(
                f"Factuality query for prompt {query.pair.prompt_id!r} is missing "
                "a gold_answer; curated prompts should be filtered out before "
                "building factuality queries."
            )
        lines.append(
            "Gold reference (for your use only — do not reveal to the responses "
            "under review):"
        )
        lines.append(query.pair.gold_answer)
        lines.append("")

    if (
        query.rubric.dimension == "calibration"
        and query.pair.source == CURATED_SOURCE
        and query.pair.expected_behavior is not None
    ):
        lines.append(
            f"For this prompt, the expected response pattern is "
            f"`{query.pair.expected_behavior}`."
        )
        lines.append("")

    lines.append("Response A:")
    lines.append(query.response_in_a)
    lines.append("")
    lines.append("Response B:")
    lines.append(query.response_in_b)
    lines.append("")
    lines.append(
        "Reason step by step about each response on this dimension, then on "
        "a final line output exactly one of: VERDICT: A, VERDICT: B, VERDICT: TIE."
    )
    return "\n".join(lines)


def _build_request_params(query: _JudgeQuery, max_tokens: int) -> dict[str, Any]:
    """Build the ``params`` dict for one batch request.

    The system block carries a 1-hour cache_control breakpoint so the
    rubric prefix is shared across every request in the batch. The user
    message is unique per request and is not cached.
    """
    return {
        "model": query.judge_model,
        "max_tokens": max_tokens,
        "system": [
            {
                "type": "text",
                "text": build_system_message(query.rubric),
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": build_user_message(query),
            }
        ],
    }


# --------------------------------------------------------------------------- #
# Disk cache
# --------------------------------------------------------------------------- #


def cache_path_for(cache_dir: Path, dimension: Dimension) -> Path:
    return cache_dir / f"judgments_{dimension}.jsonl"


def load_cache(cache_dir: Path, dimension: Dimension) -> dict[str, dict[str, Any]]:
    """Load the JSONL cache for one dimension into a ``cache_key -> record`` dict.

    If the same cache_key appears twice (legacy / concurrent-write artefact),
    the later entry wins.
    """
    path = cache_path_for(cache_dir, dimension)
    if not path.exists():
        return {}

    cache: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed cache line in %s", path)
                continue
            key = record.get("cache_key")
            if not isinstance(key, str):
                continue
            cache[key] = record
    return cache


def append_cache_record(
    cache_dir: Path, dimension: Dimension, record: Mapping[str, Any]
) -> None:
    """Append a single record to the dimension's cache JSONL."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path_for(cache_dir, dimension)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False))
        fh.write("\n")


def _record_to_raw(record: Mapping[str, Any], pair: PairToJudge, rubric: Rubric) -> _RawJudgement:
    """Reconstruct a ``_RawJudgement`` from a cached record + the live pair."""
    metadata = record.get("metadata", {}) or {}
    return _RawJudgement(
        cache_key=record["cache_key"],
        pair=pair,
        rubric_version=record.get("rubric_version", rubric.version),
        judge_model=record["judge_model"],
        position_swap=bool(metadata.get("position_swap", False)),
        verdict=_coerce_verdict(record["verdict"]),
        reasoning=record.get("reasoning", ""),
        timestamp=record.get("timestamp", ""),
    )


def _raw_to_record(raw: _RawJudgement, dimension: Dimension) -> dict[str, Any]:
    return {
        "cache_key": raw.cache_key,
        "rubric_version": raw.rubric_version,
        "judge_model": raw.judge_model,
        "verdict": raw.verdict,
        "reasoning": raw.reasoning,
        "timestamp": raw.timestamp,
        "metadata": {
            "dimension": dimension,
            "prompt_id": raw.pair.prompt_id,
            "entrant_a": raw.pair.entrant_a,
            "entrant_b": raw.pair.entrant_b,
            "position_swap": raw.position_swap,
            "source": raw.pair.source,
            "cache_creation_input_tokens": raw.cache_creation_input_tokens,
            "cache_read_input_tokens": raw.cache_read_input_tokens,
            "input_tokens": raw.input_tokens,
            "output_tokens": raw.output_tokens,
        },
    }


# --------------------------------------------------------------------------- #
# Verdict parsing & swap reconciliation
# --------------------------------------------------------------------------- #


def _coerce_verdict(value: Any) -> Verdict:
    text = str(value).strip().upper()
    if text in ("A", "B", "TIE"):
        return text  # type: ignore[return-value]
    raise ValueError(f"Unrecognised verdict value: {value!r}")


def parse_verdict(reply_text: str) -> Verdict:
    """Pull the final ``VERDICT: X`` line out of the model's reply.

    Falls back to ``TIE`` if the model emits something we cannot parse —
    callers that want stricter behaviour should inspect the reasoning
    text directly.
    """
    lines = [line.strip() for line in reply_text.splitlines() if line.strip()]
    for line in reversed(lines):
        upper = line.upper()
        if upper.startswith("VERDICT:"):
            tail = upper.split(":", 1)[1].strip()
            for candidate in ("TIE", "A", "B"):
                if tail.startswith(candidate):
                    return candidate  # type: ignore[return-value]
    logger.warning("Could not parse a verdict from judge reply; defaulting to TIE.")
    return "TIE"


def _swapped_verdict_to_entrant(verdict: Verdict, position_swap: bool, pair: PairToJudge) -> str | None:
    """Translate a position-local verdict to a concrete entrant id.

    With ``position_swap=False`` the slots map straight through;
    with ``position_swap=True`` slot A holds entrant_b and slot B holds
    entrant_a, so the entrant ids must be swapped accordingly. Ties
    return ``None``.
    """
    if verdict == "TIE":
        return None
    if not position_swap:
        return pair.entrant_a if verdict == "A" else pair.entrant_b
    return pair.entrant_b if verdict == "A" else pair.entrant_a


def reconcile_swap(
    forward: _RawJudgement, swapped: _RawJudgement, dimension: Dimension
) -> Judgment:
    """Combine the two position-ordered raw judgements into one ``Judgment``."""
    if forward.pair.prompt_id != swapped.pair.prompt_id:
        raise ValueError("Cannot reconcile raw judgements from different prompts.")
    if forward.position_swap == swapped.position_swap:
        raise ValueError(
            "Both raw judgements share the same position_swap flag; "
            "expected one forward and one swapped."
        )

    forward_winner = _swapped_verdict_to_entrant(
        forward.verdict, forward.position_swap, forward.pair
    )
    swapped_winner = _swapped_verdict_to_entrant(
        swapped.verdict, swapped.position_swap, swapped.pair
    )

    if forward_winner is not None and forward_winner == swapped_winner:
        winner = forward_winner
        inconsistent = False
    elif forward_winner is None and swapped_winner is None:
        winner = None
        inconsistent = False
    else:
        # One says A wins, the other says B wins (or one ties and the other
        # picks a side) — collapse to a tie and flag inconsistency.
        winner = None
        inconsistent = True

    pair = forward.pair
    raw_pair = (forward, swapped) if not forward.position_swap else (swapped, forward)
    return Judgment(
        prompt_id=pair.prompt_id,
        dimension=dimension,
        entrant_a=pair.entrant_a,
        entrant_b=pair.entrant_b,
        winner=winner,
        inconsistent=inconsistent,
        raw=raw_pair,
    )


# --------------------------------------------------------------------------- #
# Anthropic batch submission
# --------------------------------------------------------------------------- #


def _make_anthropic_client(config: JudgeConfig):
    """Construct an Anthropic SDK client; fail loudly if no key is configured."""
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - install error path
        raise RuntimeError(
            "The `anthropic` package is required for judge_pairs; install it "
            "via the default pixi env."
        ) from exc

    api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to your .env file (see "
            ".env.example) or pass api_key explicitly to JudgeConfig."
        )
    return anthropic.Anthropic(api_key=api_key)


def _build_batch_requests(queries: list[_JudgeQuery], max_tokens: int) -> list[dict[str, Any]]:
    """Convert each query into a Batch API request dict."""
    return [
        {
            "custom_id": query.cache_key,
            "params": _build_request_params(query, max_tokens=max_tokens),
        }
        for query in queries
    ]


def _submit_batch_with_retry(client, requests: list[dict[str, Any]], max_retries: int):
    """Submit a batch, retrying on transient rate-limit errors."""
    backoff = _SUBMIT_INITIAL_BACKOFF_SECONDS
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return client.messages.batches.create(requests=requests)
        except Exception as exc:  # pragma: no cover - transient network paths
            last_exc = exc
            status_code = getattr(exc, "status_code", None)
            if status_code != 429 or attempt == max_retries - 1:
                raise
            logger.warning(
                "Batch submission rate-limited (attempt %d/%d); sleeping %.1fs",
                attempt + 1,
                max_retries,
                backoff,
            )
            time.sleep(backoff)
            backoff *= 2
    # Defensive — the loop should either return or raise above.
    raise RuntimeError("Batch submission failed without a captured exception") from last_exc


def _poll_batch(client, batch_id: str, config: JudgeConfig) -> Any:
    """Poll until the batch ends, growing the sleep interval geometrically."""
    delay = config.poll_initial_seconds
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = getattr(batch, "processing_status", None)
        logger.info("Batch %s status: %s", batch_id, status)
        if status == "ended":
            return batch
        if status in ("canceled", "expired"):
            raise RuntimeError(
                f"Batch {batch_id} terminated with status {status!r}; "
                "see the Anthropic console for details."
            )
        time.sleep(delay)
        delay = min(config.poll_max_seconds, delay * _POLL_GROWTH)


def _extract_reply_text(message: Any) -> str:
    """Pull plain text out of a Messages API response."""
    chunks: list[str] = []
    for block in getattr(message, "content", []) or []:
        block_type = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
        if block_type == "text":
            text = getattr(block, "text", None) or (block.get("text") if isinstance(block, dict) else "")
            if text:
                chunks.append(text)
    return "\n".join(chunks)


def _process_batch_results(
    client,
    batch_id: str,
    queries_by_key: Mapping[str, _JudgeQuery],
) -> tuple[list[_RawJudgement], list[str]]:
    """Stream batch results into ``_RawJudgement`` records.

    Returns ``(judgements, failed_custom_ids)`` — failed entries are
    skipped so the caller can re-run them next time.
    """
    raw: list[_RawJudgement] = []
    failed: list[str] = []
    for entry in client.messages.batches.results(batch_id):
        custom_id = getattr(entry, "custom_id", None)
        if custom_id is None or custom_id not in queries_by_key:
            logger.warning("Batch %s returned unknown custom_id %r", batch_id, custom_id)
            continue
        query = queries_by_key[custom_id]
        result = getattr(entry, "result", None)
        result_type = getattr(result, "type", None)
        if result_type != "succeeded":
            logger.error(
                "Batch %s: request %s failed with type=%s", batch_id, custom_id, result_type
            )
            failed.append(custom_id)
            continue
        message = getattr(result, "message", None)
        reply_text = _extract_reply_text(message)
        if not reply_text:
            logger.error("Batch %s: request %s returned an empty reply", batch_id, custom_id)
            failed.append(custom_id)
            continue
        verdict = parse_verdict(reply_text)
        usage = getattr(message, "usage", None)
        raw.append(
            _RawJudgement(
                cache_key=custom_id,
                pair=query.pair,
                rubric_version=query.rubric.version,
                judge_model=query.judge_model,
                position_swap=query.position_swap,
                verdict=verdict,
                reasoning=reply_text,
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
                cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
            )
        )
    return raw, failed


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def judge_pairs(
    pairs: list[PairToJudge],
    rubric: Rubric,
    config: JudgeConfig,
    *,
    client: Any = None,
) -> JudgeResult:
    """Judge ``pairs`` on ``rubric`` and return reconciled judgments.

    Loads the on-disk cache for the rubric's dimension, filters out any
    pair-orderings already judged, submits the rest as one Anthropic
    batch with the rubric prefix marked as a 1-hour cache breakpoint,
    polls until the batch ends, writes new entries to the cache, and
    reconciles each pair's two position orders into one ``Judgment``.

    ``client`` is optional — passing one in is convenient for tests.
    """
    filtered_pairs = filter_pairs_for_dimension(pairs, rubric.dimension)
    if not filtered_pairs:
        logger.info(
            "judge_pairs(%s): no pairs to judge after filtering", rubric.dimension
        )
        return JudgeResult(judgments=[], cache_stats=CacheStats(), rubric=rubric)

    cache = load_cache(config.cache_dir, rubric.dimension)
    logger.info(
        "Loaded %d cached judgments for %s; total queries to plan: %d",
        len(cache),
        rubric.dimension,
        len(filtered_pairs) * 2,
    )

    queries: list[_JudgeQuery] = []
    cached_raw_by_key: dict[str, _RawJudgement] = {}
    for pair in filtered_pairs:
        for swap in (False, True):
            response_a = pair.response_a
            response_b = pair.response_b
            cache_key = derive_cache_key(
                rubric_text=rubric.text,
                judge_model=config.judge_model,
                prompt_id=pair.prompt_id,
                entrant_a=pair.entrant_a,
                entrant_b=pair.entrant_b,
                position_swap=swap,
                response_a=response_a,
                response_b=response_b,
            )
            query = _JudgeQuery(
                cache_key=cache_key,
                pair=pair,
                rubric=rubric,
                position_swap=swap,
                judge_model=config.judge_model,
            )
            cached_record = cache.get(cache_key)
            if cached_record is not None:
                cached_raw_by_key[cache_key] = _record_to_raw(cached_record, pair, rubric)
            else:
                queries.append(query)

    stats = CacheStats(cache_hits=len(cached_raw_by_key))
    logger.info(
        "Cache hits: %d; fresh queries to submit: %d",
        len(cached_raw_by_key),
        len(queries),
    )

    fresh_raw: list[_RawJudgement] = []
    if queries:
        if client is None:
            client = _make_anthropic_client(config)
        fresh_raw = _run_batch(queries, config, client)
        for raw in fresh_raw:
            append_cache_record(
                config.cache_dir, rubric.dimension, _raw_to_record(raw, rubric.dimension)
            )
            stats.cache_creation_input_tokens += raw.cache_creation_input_tokens
            stats.cache_read_input_tokens += raw.cache_read_input_tokens
            stats.input_tokens += raw.input_tokens
            stats.output_tokens += raw.output_tokens
            stats.fresh_calls += 1

    fresh_by_key = {raw.cache_key: raw for raw in fresh_raw}
    raw_by_key: dict[str, _RawJudgement] = {**cached_raw_by_key, **fresh_by_key}

    judgments: list[Judgment] = []
    for pair in filtered_pairs:
        forward_key = derive_cache_key(
            rubric_text=rubric.text,
            judge_model=config.judge_model,
            prompt_id=pair.prompt_id,
            entrant_a=pair.entrant_a,
            entrant_b=pair.entrant_b,
            position_swap=False,
            response_a=pair.response_a,
            response_b=pair.response_b,
        )
        swapped_key = derive_cache_key(
            rubric_text=rubric.text,
            judge_model=config.judge_model,
            prompt_id=pair.prompt_id,
            entrant_a=pair.entrant_a,
            entrant_b=pair.entrant_b,
            position_swap=True,
            response_a=pair.response_a,
            response_b=pair.response_b,
        )
        forward = raw_by_key.get(forward_key)
        swapped = raw_by_key.get(swapped_key)
        if forward is None or swapped is None:
            logger.warning(
                "Skipping reconciliation for prompt %s on %s: missing %s verdict",
                pair.prompt_id,
                rubric.dimension,
                "forward" if forward is None else "swapped",
            )
            continue
        judgments.append(reconcile_swap(forward, swapped, rubric.dimension))

    return JudgeResult(judgments=judgments, cache_stats=stats, rubric=rubric)


def _run_batch(
    queries: list[_JudgeQuery], config: JudgeConfig, client: Any
) -> list[_RawJudgement]:
    """Submit, poll, and collect one batch of judge queries."""
    requests = _build_batch_requests(queries, max_tokens=config.max_tokens)
    queries_by_key = {q.cache_key: q for q in queries}

    logger.info("Submitting batch with %d requests", len(requests))
    batch = _submit_batch_with_retry(client, requests, config.submit_max_retries)
    batch_id = getattr(batch, "id", None)
    if batch_id is None:
        raise RuntimeError("Anthropic batch creation returned no id")
    logger.info("Batch submitted: %s", batch_id)

    _poll_batch(client, batch_id, config)
    raw, failed = _process_batch_results(client, batch_id, queries_by_key)
    if failed:
        logger.error(
            "Batch %s: %d/%d requests failed; re-run to retry these.",
            batch_id,
            len(failed),
            len(requests),
        )
    return raw


# Public re-exports for tests and downstream callers.
__all__ = [
    "CURATED_SOURCE",
    "DIMENSIONS",
    "CacheStats",
    "Dimension",
    "Judgment",
    "JudgeConfig",
    "JudgeResult",
    "PairToJudge",
    "Rubric",
    "Verdict",
    "build_system_message",
    "build_user_message",
    "cache_path_for",
    "derive_cache_key",
    "filter_pairs_for_dimension",
    "judge_pairs",
    "load_cache",
    "parse_verdict",
    "reconcile_swap",
]


# Internal exports — useful for tests, not part of the stable surface.
def _make_raw_for_test(
    pair: PairToJudge,
    rubric: Rubric,
    judge_model: str,
    position_swap: bool,
    verdict: Verdict,
    reasoning: str = "",
) -> _RawJudgement:
    """Build a ``_RawJudgement`` from primitive inputs (test helper)."""
    cache_key = derive_cache_key(
        rubric_text=rubric.text,
        judge_model=judge_model,
        prompt_id=pair.prompt_id,
        entrant_a=pair.entrant_a,
        entrant_b=pair.entrant_b,
        position_swap=position_swap,
        response_a=pair.response_a,
        response_b=pair.response_b,
    )
    return _RawJudgement(
        cache_key=cache_key,
        pair=pair,
        rubric_version=rubric.version,
        judge_model=judge_model,
        position_swap=position_swap,
        verdict=verdict,
        reasoning=reasoning,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def _judgment_as_dict(judgment: Judgment) -> dict[str, Any]:
    """JSON-friendly view of a ``Judgment`` (handy for run logs)."""
    return {
        "prompt_id": judgment.prompt_id,
        "dimension": judgment.dimension,
        "entrant_a": judgment.entrant_a,
        "entrant_b": judgment.entrant_b,
        "winner": judgment.winner,
        "inconsistent": judgment.inconsistent,
        "raw": [asdict(r) for r in judgment.raw],
    }
