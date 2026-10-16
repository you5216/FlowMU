import numpy as np
import matplotlib.pyplot as plt

                               
      
                               
methods = ["Vanilla", "PURE", "SBU", "SF-NL", "Ours"]
colors = ["#F3E6F9", "#6B7B8C", "#A8B5C5", "#C4B5D6", "#C75B5B"]

attack_metrics = [
    "Origin",
    "Rewrite",
    "Induce",
]

mia_metrics = [
    "RAG",
    "Memory",
    "LLM",
    "Agent",
    "Adaptive",
]

metrics = attack_metrics + mia_metrics

data = {
    "Vanilla": [
        90.00, 96.50, 93.50,
        96.24, 52.59, 59.02, 55.72, 88.51,
    ],
    "PURE": [
        53.00, 75.00, 77.50,
        67.97, 88.74, 63.96, 84.96, 93.25,
    ],
    "SBU": [
        63.50, 79.00, 78.50,
        75.52, 99.22, 69.32, 90.06, 90.99,
    ],
    "SF-NL": [
        78.00, 97.50, 92.00,
        96.24, 88.74, 81.35, 51.76, 95.21,
    ],
    "Ours": [
        28.50, 43.00, 42.00,
        61.09, 54.55, 60.14, 51.28, 58.07,
    ],
}

                               
               
                               
x = np.arange(len(metrics))
num_methods = len(methods)
group_width = 0.7
bar_width = group_width / num_methods

fig, ax = plt.subplots(figsize=(9, 5))
ax.set_facecolor("#F0F0F7")

ax2 = ax.twinx()
ax2.set_facecolor("#F0F0F7")

for i, method in enumerate(methods):
    offset = (i - (num_methods - 1) / 2) * bar_width

    bars1 = ax.bar(
        x[:3] + offset,
        data[method][:3],
        width=bar_width,
        label=method,
        color=colors[i],
        edgecolor="black",
        linewidth=1.2 if method == "Ours" else 0.5,
        hatch="//" if method == "Ours" else None,
        zorder=3,
    )

    bars2 = ax2.bar(
        x[3:] + offset,
        data[method][3:],
        width=bar_width,
        color=colors[i],
        edgecolor="black",
        linewidth=1.2 if method == "Ours" else 0.5,
        hatch="//" if method == "Ours" else None,
        zorder=3,
    )

    for bar, value in zip(bars1, data[method][:3]):
        if method == "Ours":
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.0,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                rotation=90,
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() - 1.0,
                f"{value:.2f}",
                ha="center",
                va="top",
                fontsize=10,
                rotation=90,
            )

    for bar, value in zip(bars2, data[method][3:]):
        if method == "Ours":
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.0,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                rotation=90,
            )
        else:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() - 1.0,
                f"{value:.2f}",
                ha="center",
                va="top",
                fontsize=10,
                rotation=90,
            )

                               
                   
                               
                                          
ax.axvline(
    len(attack_metrics) - 0.5,
    linestyle="--",
    linewidth=2,
    color="gray",
)

ax.text(
    1,
    104,
    "(a) Recovery Attacks",
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold",
)

ax.text(
    5,
    104,
    "(b) Membership Inference Attacks",
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold",
)

                               
                 
                               
ax.set_ylabel("Recovery ACC (%)", fontsize=15)
ax2.set_ylabel("MIA AUC (%)", fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels(metrics, ha="center", fontsize=15)
ax.set_ylim(15, 108)
ax2.set_ylim(15, 108)
ax.set_xlim(-0.6, len(metrics) - 0.4)

ax.tick_params(axis="y", labelsize=14)
ax2.tick_params(axis="y", labelsize=14)

ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)
ax.legend(
    ncol=5,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.13),
    frameon=True,
    fontsize=15,
)

fig.tight_layout()
ax.yaxis.set_label_coords(-0.06, 0.5)
ax2.yaxis.set_label_coords(1.06, 0.5)

                                  
plt.savefig(
    "recovery_attack_tofu_for05_allmethods.pdf",
    bbox_inches="tight",
)

plt.show()