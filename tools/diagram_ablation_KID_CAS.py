# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置论文友好风格（和你提供的图一致）
mpl.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
})

# 柔和淡色调色板（与原图一致）
colors = [
    "#c6dbef",  # 浅蓝
    "#b2e2e2",  # 浅青绿
    "#fdae6b",  # 浅橙
    "#cccccc",  # 浅灰
]

# =========================
# 数据
# =========================
variants = ["LoRA", "LoRA+TI", "LoRA+PA", "LoRA+TI+PA"]
categories = ["Algal Leaf Spot", "Blight Leaf", "Leaf Spot", "No Disease"]

kid_data = {
    "LoRA":        [0.010, 0.020, 0.020, 0.030],
    "LoRA+TI":     [0.010, 0.020, 0.018, 0.028],
    "LoRA+PA":     [0.010, 0.015, 0.017, 0.027],
    "LoRA+TI+PA":  [0.009, 0.010, 0.016, 0.026],
}

cas_data = {
    "LoRA":        [0.975, 0.950, 0.825, 0.975],
    "LoRA+TI":     [0.980, 0.955, 0.835, 0.970],
    "LoRA+PA":     [0.982, 0.957, 0.840, 0.972],
    "LoRA+TI+PA":  [0.985, 0.960, 0.850, 0.973],
}

# =========================
# 绘制柱状图
# =========================
x = np.arange(len(categories))
bar_w = 0.18
offsets = np.linspace(-1.5, 1.5, len(variants)) * bar_w

fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

# ---- (a) KID ----
ax = axes[0]
for i, v in enumerate(variants):
    ax.bar(
        x + offsets[i], kid_data[v], width=bar_w,
        label=v, color=colors[i], edgecolor="black", linewidth=0.6
    )
ax.set_title("(a) KID Score")
ax.set_ylabel("KID (↓)")
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=15, ha="right")
ax.grid(axis="y", linestyle="--", alpha=0.3)

# ---- (b) CAS ----
ax = axes[1]
for i, v in enumerate(variants):
    ax.bar(
        x + offsets[i], cas_data[v], width=bar_w,
        label=v, color=colors[i], edgecolor="black", linewidth=0.6
    )
ax.set_title("(b) CAS (Top-1)")
ax.set_ylabel("CAS (↑)")
ax.set_ylim(0.75, 1.00)
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=15, ha="right")
ax.grid(axis="y", linestyle="--", alpha=0.3)

# 图例：底部居中
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(variants), frameon=False)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig("Fig_ablation_KID_CAS_matched.png", bbox_inches="tight")
plt.show()

print("Saved: Fig_ablation_KID_CAS_matched.png")
