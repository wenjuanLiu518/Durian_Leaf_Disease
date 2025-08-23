import matplotlib.pyplot as plt
import numpy as np

# test_acc数据定义
categories = ['Algal Spot', 'Blight', 'Leaf Spot', 'No disease']
#real dataset test_acc
real_values = [0.74, 0.57, 0.12, 0.74]
#(real + synthetic) dataset test_acc
synthetic_values = [0.89, 0.88, 0.51, 0.69]

# 图示设置
x = np.arange(len(categories))
width = 0.35

# # Fig. 4a: 柱形图
# plt.figure(figsize=(10, 6))
# plt.bar(x - width/2, real_values, width, label='Real', color='skyblue')
# plt.bar(x + width/2, synthetic_values, width, label='Real + Synthetic', color='lightgreen')
# plt.ylabel('test_acc')
# plt.title('test_acc')
# plt.xticks(x, categories, rotation=45, ha='right')
# plt.ylim(0, 1)
# plt.legend()
# plt.tight_layout()
# plt.savefig('fig_4a.png')
# plt.close()

# Fig. 4b: 折线图（修改后）
plt.figure(figsize=(10, 6))
ax = plt.gca()
ax.plot(x, real_values, marker='o', label='Real', color='blue')
ax.plot(x, synthetic_values, marker='s', label='Real + Synthetic', color='red')
ax.set_ylabel('test_acc')
ax.set_title('test_acc comparison')
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=0, ha='center')
ax.set_yticks(np.arange(0, 1.1, 0.2))  # Y 轴刻度从 0 到 1，步长 0.2
ax.set_ylim(0, 1)
ax.spines['top'].set_visible(False)  # 隐藏顶部轴线
ax.spines['right'].set_visible(False)  # 隐藏右侧轴线
ax.spines['left'].set_visible(True)  # 保留左侧轴线
ax.spines['bottom'].set_visible(True)  # 保留底部轴线
ax.tick_params(axis='both', which='both', direction='in', length=6)  # 刻度线向内，长度为 6
ax.legend()
plt.tight_layout()
plt.savefig('Fig10.tif', format='tif', dpi=600, pil_kwargs={'compression': 'tiff_lzw'}, bbox_inches='tight')
plt.close()

# # Fig. 4c: 条形图（水平）
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.barh(x - width/2, real_values, width, label='Real', color='skyblue')
# ax.barh(x + width/2, synthetic_values, width, label='Real + Synthetic', color='lightgreen')
# ax.set_xlabel('Test Accuracy')
# ax.set_title('Test Accuracy Comparison (Fig. 4c)')
# ax.set_yticks(x)
# ax.set_yticklabels(categories)
# ax.set_xlim(0, 1)
# ax.legend()
# plt.tight_layout()
# plt.savefig('fig_4c.png', format='tif', dpi=600, pil_kwargs={'compression': 'tiff_lzw'}, bbox_inches='tight')
# plt.close()