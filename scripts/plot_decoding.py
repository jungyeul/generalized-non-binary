"""Plot mean single-sentence decoding latency by corpus and token band."""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CORPORA = (("ptb", "PTB"), ("ctb", "CTB"))
ORDERS = (("td", "Top-down"), ("io", "In-order"), ("bu", "Bottom-up"))
SYSTEMS = (
    ("canonical", "Canonical", "#0072B2", "o", "-", -0.045),
    ("generalised", "Delimiter-guided", "#D55E00", "s", "--", 0.045),
)
BANDS = ("1–10", "11–20", "21–30", "31–40", "41+")


def latency_series(groups, key):
    values = groups[key]["latency_ms"]
    if len(values) != len(BANDS):
        raise ValueError(f"{key} must contain {len(BANDS)} sentence-length bands")
    means = [float(value["mean"]) for value in values]
    deviations = [float(value["sample_sd"]) for value in values]
    if any(not math.isfinite(value) or value < 0 for value in means + deviations):
        raise ValueError(f"{key} contains an invalid latency or sample SD")
    return means, deviations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="JSON containing benchmark groups")
    parser.add_argument("output", type=Path, help="Output PNG path")
    args = parser.parse_args()

    data = json.loads(args.summary.read_text(encoding="utf-8"))
    groups = data.get("groups", data)
    plt.rcParams.update(
        {"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 9,
         "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.15), sharex=True, sharey="row")
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.155, top=0.83, wspace=0.13, hspace=0.30)
    for row, (corpus, corpus_label) in enumerate(CORPORA):
        whiskers = []
        for col, (order, order_label) in enumerate(ORDERS):
            ax = axes[row, col]
            for system, label, color, marker, linestyle, offset in SYSTEMS:
                means, deviations = latency_series(groups, f"{system}-{order}-{corpus}")
                whiskers.extend(mean - deviation for mean, deviation in zip(means, deviations))
                ax.errorbar(
                    [i + offset for i in range(len(BANDS))], means, yerr=deviations,
                    label=label, color=color, marker=marker, linestyle=linestyle,
                    linewidth=1.35, markersize=4.0, capsize=2.5, elinewidth=0.85,
                )
            ax.set_xticks(range(len(BANDS)), BANDS)
            ax.set_xlim(-0.35, len(BANDS) - 0.65)
            ax.grid(axis="y", color="#dddddd", linewidth=0.55)
            ax.set_axisbelow(True)
            ax.tick_params(axis="both", length=2.6)
            if row == 0:
                ax.set_title(order_label, pad=7)
        axes[row, 0].set_ylabel(f"{corpus_label}\nDecoding time (ms)")
        axes[row, 0].set_ylim(bottom=min(0.0, min(whiskers)))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.55, 0.98),
               ncols=2, frameon=False)
    fig.supxlabel("Sentence length (tokens)", x=0.55, y=0.055, fontsize=9)
    fig.savefig(args.output, dpi=450, format="png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
