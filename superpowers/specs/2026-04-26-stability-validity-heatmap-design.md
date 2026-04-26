# Stability Radius & Validity Heatmap — Design Spec

**Date:** 2026-04-26
**Branch:** feat/validity-stability

---

## Context

TAP's PoE inference already surfaces a token-entropy heatmap (verifier ensemble predictive entropy per emitted token). This is a soft security signal. The goal is to add two formal metrics grounded in the TPA paper (Ghitu & Wicker, 2025, arXiv:2602.09757):

1. **Stability radius** — per-token certified lower bound on how many verifier heads an adversary must corrupt to flip the accepted token to anything else. Directly mirrors TPA's Definition 4.1 and the DPA vote-margin formula.
2. **Validity radius** — at rejection positions only, how many heads must be corrupted to force the specific rejected draft token through. Mirrors TPA's Definition 4.2, computed via a greedy simulation (exact for our inference-time adversary; Algorithm 1 from the paper is a lower bound and fails on tied-competitor inputs).

Both are displayed in a new `StabilityHeatmapPanel` rendered below the existing entropy panel.

---

## Metrics

### Stability radius (every emitted token)

**Integer (0–4 for 8 verifier heads):**
```
per_head_winners = argmax(v_logits, dim=-1)   # shape (n_verifiers,)
n_A = count of heads voting for accepted token
n_B = max count for any single alternative token   # torch.unique, tight bound
stability_radius = (n_A - n_B) // 2
```

**Continuous margin (heatmap colour):**
```
top2 = topk(P_dist, 2)
stability_margin = top2[0] - top2[1]           # float in [0, 1]
```

### Validity radius (rejection positions only)

Greedy simulation: repeatedly flip a vote from the current highest-competing head to the target (rejected draft token) until it wins plurality. Count = minimum attack budget = validity radius per Definition 4.2.

```python
def _validity_radius(per_head_winners, target_id):
    counts = Counter(per_head_winners.tolist())
    n_target = counts.pop(target_id, 0)
    competitors = sorted(counts.values(), reverse=True)
    k = 0
    while competitors and n_target <= competitors[0]:
        competitors[0] -= 1
        competitors.sort(reverse=True)
        n_target += 1
        k += 1
    return k
```

Algorithm 1 from the paper is NOT used because it gives incorrect results when competitors are tied (Δ = 0 case), underestimating the attack budget. The greedy is exact under the direct head-flip adversary model appropriate for inference-time ensembles.

### Suppression score (rejection positions only)

```
suppression_score = P_dist[rejected_draft_token]   # float in [0, 1]
```

Continuous analogue of validity: how much probability mass the ensemble assigned to the rejected token.

---

## Architecture

### `olmo_tap/inference/poe.py`

Replace the 5-element return tuple of `generate_with_cache` with a `PoEOutput` dataclass:

```python
@dataclass
class PoEOutput:
    output_parts:       list[str]       # output_parts[0] = prompt prefix; [1:] = emitted tokens
    original_tokens:    list[str]
    resampled_idxs:     list[int]
    token_entropies:    list[float]
    uncertainty:        float | None
    stability_radii:    list[int]       # parallel to output_parts[1:]
    stability_margins:  list[float]     # parallel to output_parts[1:]
    validity_radii:     list[int]       # parallel to resampled_idxs
    suppression_scores: list[float]     # parallel to resampled_idxs
```

Compute stability fields at every accept/reject step. Compute validity and suppression fields only at rejection steps.

### `app/backend/hydra_inference.py`

- `generate()` unpacks `PoEOutput` and passes new fields onward
- `_tokens_and_resamples_from_poe_output()` adds `validity_radius` and `suppression_score` to each `resampled` dict; `severity` remains hardcoded to `1.0`

### `app/backend/response_payloads.py`

`poe_security()` extended with `stability_radii` and `stability_margins` parameters. New security payload shape:

```python
{
    "certified":          True,
    "tokens":             list[str],
    "resampled":          list[{
        "index":             int,
        "old_token":         str,
        "new_token":         str,
        "severity":          1.0,        # hardcoded, unchanged
        "validity_radius":   int,
        "suppression_score": float,
    }],
    "token_entropies":    list[float],
    "stability_radii":    list[int],
    "stability_margins":  list[float],
}
```

### `app/frontend/src/lib/api.ts`

```typescript
interface SecurityResample {
  index:              number;
  old_token:          string;
  new_token:          string;
  severity:           number;
  validity_radius?:   number;
  suppression_score?: number;
}

interface SecurityStatus {
  certified:           boolean | null;
  tokens:              string[];
  resampled:           SecurityResample[];
  token_entropies?:    number[];
  stability_radii?:    number[];
  stability_margins?:  number[];
}
```

### `app/frontend/src/components/StabilityHeatmapPanel.tsx` (new file)

Mirrors `TokenHeatmapPanel` structure. Renders below the entropy panel only when `stability_radii` and `stability_margins` are present in the security payload.

- Token chip colour: `rgba(--color-accent-rgb, (1 - stability_margin) * 0.55)` — low margin (contested) → bright
- Rejection positions: small visual marker (underline or dot)
- Tooltip on every token: stability radius integer
- Tooltip at rejection positions: additionally shows suppression score and `old_token`

### `app/frontend/src/components/SecurityTokensPanel.tsx` (modified)

The existing `ResampledToken` tooltip currently shows `old_token → new_token`. Extend it to also display the validity radius:

```tsx
tooltipBody={
  <>
    {resample.old_token} → {resample.new_token}
    {resample.validity_radius != null && (
      <> · validity radius: {resample.validity_radius}</>
    )}
  </>
}
```

---

## Data flow

```
PoE.generate_with_cache()
  → PoEOutput (stability_radii, stability_margins, validity_radii, suppression_scores)
  → hydra_inference.generate()
  → poe_security(tokens, resampled, token_entropies, stability_radii, stability_margins)
  → /api/analyse JSON response
  → StabilityHeatmapPanel (frontend)
```

---

## Relationship to TPA paper

| Paper concept | TAP implementation |
|---|---|
| Stability radius (Def 4.1) | `stability_radius = (n_A - n_B) // 2` per token |
| Validity radius (Def 4.2) | Greedy on per-head votes at rejection positions |
| Algorithm 1 (TPA) | Not used — fails on tied competitors; greedy is exact |
| Continuous margin | `stability_margin = P_dist[top1] - P_dist[top2]` |
| Suppression | `P_dist[rejected_token]` |

The adversary model differs from TPA: TPA counts training-data modifications (weaker adversary, data-poisoning); TAP counts direct head-vote flips (stronger adversary, inference-time). The greedy gives the exact minimum budget under our model.

---

## Verification

1. Run `pixi run -e cuda pytest` — existing tests must pass
2. Start the dev server and send a prompt through `/api/analyse`
3. Confirm `security` payload in response contains `stability_radii` and `stability_margins`
4. Confirm each `resampled` entry contains `validity_radius` and `suppression_score`
5. Confirm `StabilityHeatmapPanel` renders below the entropy panel
6. Confirm token colours vary with `stability_margin`, tooltips show integer radius
7. Confirm rejection position markers and tooltip additional fields appear correctly
8. Test with HF fallback (`hf=true`) — stability fields absent, panel does not render
