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


def plot_results(results, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax_ttft, ax_latency, ax_tps = axes

    plot_histogram_kde(
        ax_ttft, results["hydra"]["ttft"]["filtered_ms"], "Hydra", "tab:blue"
    )
    if "baseline" in results:
        plot_histogram_kde(
            ax_ttft,
            results["baseline"]["ttft"]["filtered_ms"],
            "Baseline",
            "tab:orange",
        )
    ax_ttft.set_title("TTFT Distribution")
    ax_ttft.legend()

    plot_decode_curve(
        ax_latency, ax_tps, results["hydra"]["decode"], "Hydra", "tab:blue"
    )
    if "baseline" in results:
        plot_decode_curve(
            ax_latency, ax_tps, results["baseline"]["decode"], "Baseline", "tab:orange"
        )
    ax_latency.set_title("Decode Latency vs Position")
    ax_latency.legend()
    ax_tps.set_title("TPS vs Position")
    ax_tps.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "graph.png", dpi=150)
    plt.close(fig)
    print(f"Plot saved to {output_dir / 'graph.png'}")
