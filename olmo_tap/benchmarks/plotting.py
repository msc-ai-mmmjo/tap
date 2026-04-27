"""
Plot the three-config benchmark output produced by
:mod:`olmo_tap.benchmarks.inference`.

A single figure with three subplots:

1. **TTFT distribution**     — KDE + histogram of prefill (time-to-first-token)
   timings for ``baseline`` and ``hydra_naive``. PoE is excluded because
   prefill is bundled inside its full-generation call.
2. **Decode latency vs KV position** — per-step decode latency (ms) over the
   benchmarked KV positions. PoE is rendered as horizontal dashed lines, one
   per γ, at its **per-token equivalent latency** (call median /
   accepted tokens) so it sits on the same axis as the per-step rows.
3. **TPS vs KV position**    — tokens per second, same shape. PoE rendered as
   horizontal dashed lines at its effective TPS per γ.

Colour convention: orange = baseline, blue = naive Hydra, green family = PoE
(darker greens for larger γ).

Usually invoked indirectly — :func:`olmo_tap.benchmarks.inference.main` calls
:func:`plot_results` after writing ``results.json``.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def plot_histogram_kde(ax, timings, label, color):
    data = np.array(timings)
    ax.hist(data, bins=30, alpha=0.3, color=color, density=True, label=f"{label} hist")
    kde = gaussian_kde(data)
    x = np.linspace(data.min(), data.max(), 200)
    ax.plot(x, kde(x), color=color, label=f"{label} KDE")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Density")


def plot_decode_curve(ax_latency, ax_tps, decode_results, label, color):
    positions = sorted(decode_results["per_position"].keys(), key=int)
    medians = [decode_results["per_position"][p]["median_ms"] for p in positions]
    p20s = [decode_results["per_position"][p]["p20_ms"] for p in positions]
    p80s = [decode_results["per_position"][p]["p80_ms"] for p in positions]
    tps = [decode_results["per_position"][p]["tps"] for p in positions]
    pos_ints = [int(p) for p in positions]

    ax_latency.plot(pos_ints, medians, color=color, label=label)
    ax_latency.fill_between(pos_ints, p20s, p80s, alpha=0.2, color=color)
    ax_latency.set_xlabel("KV Cache Position")
    ax_latency.set_ylabel("Latency (ms)")

    ax_tps.plot(pos_ints, tps, color=color, label=label)
    ax_tps.set_xlabel("KV Cache Position")
    ax_tps.set_ylabel("Tokens/sec")


PALETTE = {
    "baseline": "tab:orange",
    "hydra_naive": "tab:blue",
    "hydra_poe": "tab:green",
}

LABELS = {
    "baseline": "Baseline OLMo",
    "hydra_naive": "Hydra (naive avg)",
    "hydra_poe": "Hydra + PoE",
}


def plot_results(results, output_dir):
    """Render the three-config comparison figure to ``graph.png``.

    :param results: Benchmark results dict as written to ``results.json`` by
        :func:`olmo_tap.benchmarks.inference.main`. Recognised top-level keys:
        ``baseline``, ``hydra_naive``, ``hydra_poe``. Missing keys are
        skipped silently — a partial run still plots.
    :param output_dir: Directory to write ``graph.png`` into. Must exist.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax_ttft, ax_latency, ax_tps = axes

    # TTFT distribution + per-position decode curve for the two non-PoE configs
    for key in ("baseline", "hydra_naive"):
        if key not in results:
            continue
        color = PALETTE[key]
        label = LABELS[key]
        plot_histogram_kde(ax_ttft, results[key]["ttft"]["filtered_ms"], label, color)
        plot_decode_curve(ax_latency, ax_tps, results[key]["decode"], label, color)

    # PoE prefill TTFT (residual_forward over all heads, no draft loop).
    if "hydra_poe" in results and "ttft" in results["hydra_poe"]:
        plot_histogram_kde(
            ax_ttft,
            results["hydra_poe"]["ttft"]["filtered_ms"],
            f"{LABELS['hydra_poe']} (prefill)",
            PALETTE["hydra_poe"],
        )

    # PoE: per-gamma effective TPS + per-token equivalent latency. Per-token
    # latency = call_median_ms / avg_accepted_tokens (apples-to-apples vs the
    # other rows' per-step decode latency).
    if "hydra_poe" in results and "per_gamma" in results["hydra_poe"]:
        per_gamma = results["hydra_poe"]["per_gamma"]
        cmap = plt.get_cmap("Greens")
        n = len(per_gamma)
        for i, (g_str, p) in enumerate(
            sorted(per_gamma.items(), key=lambda kv: int(kv[0]))
        ):
            color = cmap(0.4 + 0.5 * (i / max(n - 1, 1)))
            per_tok_ms = (
                p["median_ms"] / p["avg_accepted_tokens_per_call"]
                if p["avg_accepted_tokens_per_call"] > 0
                else 0.0
            )
            label_g = f"PoE γ={g_str}"
            ax_tps.axhline(
                p["effective_tps"],
                color=color,
                linestyle="--",
                label=f"{label_g} ({p['effective_tps']} tok/s, resample={p['resample_rate']})",
            )
            ax_latency.axhline(
                per_tok_ms,
                color=color,
                linestyle="--",
                label=f"{label_g} ({per_tok_ms:.1f} ms/tok eq.)",
            )

    ax_ttft.set_title("TTFT Distribution")
    ax_ttft.legend()
    ax_latency.set_title("Decode Latency vs Position")
    ax_latency.legend()
    ax_tps.set_title("TPS vs Position")
    ax_tps.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "graph.png", dpi=150)
    plt.close(fig)
    print(f"Plot saved to {output_dir / 'graph.png'}")
