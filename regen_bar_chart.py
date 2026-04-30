import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Correct values from actual experimental data
# ---------------------------------------------------------------------------
means = [0.6016, 0.6488, 0.7007, 0.7197]
ci_lo = [0.579,  0.623,  0.673,  0.702 ]
ci_hi = [0.623,  0.673,  0.728,  0.739 ]

yerr_lo = [m - lo for m, lo in zip(means, ci_lo)]
yerr_hi = [hi - m  for m, hi in zip(means, ci_hi)]

labels = [
    "Llama-3.1-8B\nUnconstrained",
    "Llama-3.1-8B\nConstrained",
    "GPT-OSS-120B\nUnconstrained",
    "GPT-OSS-120B\nConstrained",
]

# ---------------------------------------------------------------------------
# Colours: two hues (one per model), lighter shade = unconstrained
# ---------------------------------------------------------------------------
LLAMA_LIGHT  = "#7bafd4"
LLAMA_DARK   = "#2166ac"
GPT_LIGHT    = "#f4a582"
GPT_DARK     = "#d6604d"

colors = [LLAMA_LIGHT, LLAMA_DARK, GPT_LIGHT, GPT_DARK]

# ---------------------------------------------------------------------------
# Figure — single-column ACM width (3.33 in), doubled for readability
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.2))
fig.patch.set_facecolor("white")

x = np.arange(len(labels))
bars = ax.bar(
    x, means,
    color=colors,
    width=0.58,
    yerr=[yerr_lo, yerr_hi],
    capsize=4,
    error_kw={"elinewidth": 1.2, "ecolor": "#333333", "capthick": 1.2},
    zorder=3,
)

# Value annotations
for bar, val in zip(bars, means):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(yerr_hi) * 0.08 + 0.012,
        f"{val:.3f}",
        ha="center", va="bottom",
        fontsize=8.5, fontweight="bold", color="#222222",
    )

# Reference line at J = 0.5
ax.axhline(0.5, color="#888888", linestyle="--", linewidth=0.9, zorder=2)
ax.text(3.45, 0.502, "$J = 0.5$", va="bottom", ha="right",
        fontsize=7.5, color="#888888")

# Axes
ax.set_ylim(0, 0.92)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel("Mean Row Jaccard Similarity", fontsize=9.5)
ax.set_title("Output Stability Across Models and Conditions",
             fontsize=10.5, fontweight="bold", pad=10)

ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.6, zorder=0)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="both", labelsize=8.5)

# Legend
legend_patches = [
    mpatches.Patch(color=LLAMA_LIGHT, label="Llama-3.1-8B  Unconstrained"),
    mpatches.Patch(color=LLAMA_DARK,  label="Llama-3.1-8B  Constrained"),
    mpatches.Patch(color=GPT_LIGHT,   label="GPT-OSS-120B  Unconstrained"),
    mpatches.Patch(color=GPT_DARK,    label="GPT-OSS-120B  Constrained"),
]
ax.legend(
    handles=legend_patches,
    fontsize=7.5, framealpha=0.9,
    loc="upper left", borderpad=0.7,
    handlelength=1.2, handleheight=0.9,
)

plt.tight_layout(pad=0.8)

for path in [
    "results/stability_bar_chart.png",
    "paper/figures/stability_bar.png",
]:
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved → {path}")

# Also save PDF for LaTeX
plt.savefig("paper/figures/stability_bar.pdf", bbox_inches="tight")
print("Saved → paper/figures/stability_bar.pdf")
