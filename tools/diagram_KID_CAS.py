import matplotlib.pyplot as plt
import numpy as np

# 设置样式以减少默认边框
plt.style.use('seaborn-v0_8-white')

# 数据
categories = ['Algal Leaf Spot', 'Leaf Blight', 'Leaf Spot', 'No Disease']
methods = ['FLUX', 'SD3.5', 'SDXL']

# KID 数据
kid_means = {
    'FLUX': [0.01, 0.02, 0.02, 0.03],
    'SD3.5': [0.05, 0.04, 0.01, 0.03],
    'SDXL': [0.07, 0.06, 0.02, 0.03]
}

# CAS 数据
cas_scores = {
    'FLUX': [0.975, 0.950, 0.825, 0.975],
    'SD3.5': [0.775, 0.875, 0.575, 0.650],
    'SDXL': [0.750, 0.925, 0.750, 0.925]
}

# 颜色
colors = {
    'FLUX': '#FFC1CC',
    'SD3.5': '#87CEEB',
    'SDXL': '#00008B'
}

# 条形图位置参数
x = np.arange(len(categories))
width = 0.25

# 图1：KID 分数
fig1, ax1 = plt.subplots(figsize=(8, 6))
bars1 = []
for i, method in enumerate(methods):
    bars = ax1.bar(x + i * width, kid_means[method], width, label=method, color=colors[method], edgecolor=colors[method], linewidth=0)
    bars1.extend(bars)
for bar in bars1:
    bar.set_edgecolor(bar.get_facecolor())
    bar.set_linewidth(0)

# 自定义 KID 图
ax1.set_ylabel('KID (mean)')
ax1.set_title('(a)                                                             KID Score (mean)', fontweight='bold',loc = 'left')
ax1.set_xticks(x + width)
ax1.set_xticklabels(categories)
ax1.set_ylim(0, 0.08)
ax1.set_yticks(np.arange(0, 0.09, 0.01))
ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
ax1.tick_params(axis='both', which='major', length=5)
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax1.grid(False)

# 保留底部和左侧轴线，隐藏顶部和右侧
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(True)
ax1.spines['left'].set_visible(True)

# 调整布局并保存为 TIFF 格式，600 DPI
plt.tight_layout()
plt.savefig('kid_scores_plot.tif', format='tif', dpi=600, pil_kwargs={'compression': 'tiff_lzw'}, bbox_inches='tight')
from PIL import Image
img = Image.open('kid_scores_plot.tif')
print(f"Compression: {img.info.get('compression')}")
print(f"DPI: {img.info.get('dpi')}")
print(f"DPI: {img.info.get('dpi')}")
plt.close(fig1)

# 图2：CAS 分数
fig2, ax2 = plt.subplots(figsize=(8, 6))
bars2 = []
for i, method in enumerate(methods):
    bars = ax2.bar(x + i * width, cas_scores[method], width, label=method, color=colors[method], edgecolor=colors[method], linewidth=0)
    bars2.extend(bars)
for bar in bars2:
    bar.set_edgecolor(bar.get_facecolor())
    bar.set_linewidth(0)

# 自定义 CAS 图
ax2.set_ylabel('CAS (top1)')
ax2.set_title('(b)                                                            CAS (top1)', fontweight='bold', loc= 'left')
ax2.set_xticks(x + width)
ax2.set_xticklabels(categories)
ax2.set_ylim(0, 1.0)
ax2.set_yticks(np.arange(0, 1.01, 0.1))
ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
ax2.tick_params(axis='both', which='major', length=5)
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax2.grid(False)

# 保留底部和左侧轴线，隐藏顶部和右侧
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(True)
ax2.spines['left'].set_visible(True)

# 调整布局并保存为 TIFF 格式，600 DPI
plt.tight_layout()
plt.savefig('cas_scores_plot.tif', format='tif', dpi=600, pil_kwargs={'compression': 'tiff_lzw'}, bbox_inches='tight')
plt.close(fig2)