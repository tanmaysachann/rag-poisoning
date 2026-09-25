"""Build presentation-ready figures from the sealed frozen test snapshot."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation.verify_test_v1 import ROOT, verify


def main() -> None:
    verify(ROOT)
    data = json.loads((ROOT / "results/test_intervals_v1.json").read_text(encoding="utf-8"))
    labels = ["Greedy insertion", "Stealth substitution"]
    profiles = [data["profiles"]["greedy_minilm_ranker"],
                data["profiles"]["stealth_minilm_ranker"]]
    x = np.arange(2)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.24, top=0.79, wspace=0.18)

    def bars(ax, keys, colors, legend):
        width = 0.36
        for index, (key, color, title) in enumerate(zip(keys, colors, legend)):
            offset = (index - 0.5) * width
            values = [profile[key]["rate"] for profile in profiles]
            lows = [profile[key]["wilson_95"][0] for profile in profiles]
            highs = [profile[key]["wilson_95"][1] for profile in profiles]
            ax.bar(x + offset, values, width=width, color=color, label=title, zorder=2)
            ax.errorbar(x + offset, values,
                        yerr=[np.asarray(values) - lows, np.asarray(highs) - values],
                        fmt="none", color="#263238", capsize=3, linewidth=1, zorder=3)
            for pos, high, profile in zip(x + offset, highs, profiles):
                ax.text(pos, min(high + 0.025, 1.06),
                        f"{profile[key]['count']}/75", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1.12)
        ax.set_yticks(np.linspace(0, 1, 6), [f"{int(v * 100)}%" for v in np.linspace(0, 1, 6)])
        ax.grid(axis="y", color="#dfe5e8", linewidth=0.7, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="upper right", fontsize=8)

    bars(axes[0], ["undefended_attack_success", "defended_attack_success"],
         ["#c05848", "#236e8f"], ["Before defense", "After defense"])
    axes[0].set_title("Attacker answer success")
    bars(axes[1], ["attack_quarantine", "clean_alias_recovery"],
         ["#738da6", "#2d846f"], ["Attack quarantined", "Clean alias recovered"])
    axes[1].set_title("Defense decision and answer utility")
    fig.suptitle("Frozen test v1 · accepted-ingest attacks · MiniLM + sentence ranker", y=0.97, fontsize=12)
    fig.text(0.5, 0.035,
             "75 query/passage pairs per family; error bars show Wilson 95% intervals. "
             "Known-alias recovery is a narrow answer-quality measure.",
             ha="center", va="bottom", fontsize=8, color="#455a64")
    out = ROOT / "output/research_figures"
    out.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        path = out / f"test_security_v1.{extension}"
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
