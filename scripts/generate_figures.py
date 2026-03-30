"""生成 README 用的可视化图表"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs("assets", exist_ok=True)

plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'figure.facecolor': 'white',
    'axes.facecolor': '#FAFAFA',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# ============================================================
# 图1: 压缩质量 vs Bits
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

bits = [2, 3, 4]
cosine_sims = [0.9388, 0.9822, 0.9942]
compression_ratios = [15.2, 10.0, 7.3]

# 左图: 余弦相似度
ax = axes[0]
bars = ax.bar(bits, cosine_sims, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], width=0.6, edgecolor='white', linewidth=2)
ax.set_xlabel('Quantization Bits')
ax.set_ylabel('Cosine Similarity')
ax.set_title('Compression Quality on Wan2.1 Features')
ax.set_ylim(0.90, 1.01)
ax.set_xticks(bits)
ax.set_xticklabels(['2-bit', '3-bit', '4-bit'])
ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='Quality threshold')
for bar, val in zip(bars, cosine_sims):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
ax.legend()

# 右图: 压缩比
ax = axes[1]
bars = ax.bar(bits, compression_ratios, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], width=0.6, edgecolor='white', linewidth=2)
ax.set_xlabel('Quantization Bits')
ax.set_ylabel('Compression Ratio')
ax.set_title('Cache Size Reduction')
ax.set_xticks(bits)
ax.set_xticklabels(['2-bit', '3-bit', '4-bit'])
for bar, val in zip(bars, compression_ratios):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.1f}×', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('assets/compression_quality.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ assets/compression_quality.png")


# ============================================================
# 图2: 旋转高斯化效果
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 模拟旋转前的分布（高峰度）
np.random.seed(42)
d = 1536
x_before = np.random.randn(d) * 0.1
x_before[np.random.choice(d, 15)] = np.random.randn(15) * 1.5
x_before = x_before / np.linalg.norm(x_before)

# 旋转后的分布（高斯）
R = np.random.randn(d, d)
Q, _ = np.linalg.qr(R)
x_after = Q @ x_before

ax = axes[0]
ax.hist(x_before, bins=80, color='#FF6B6B', alpha=0.8, edgecolor='white', linewidth=0.5)
ax.set_title(f'Before Rotation (kurtosis={float(np.mean(x_before**4)/np.mean(x_before**2)**2 - 3):.1f})')
ax.set_xlabel('Coordinate Value')
ax.set_ylabel('Count')
ax.set_xlim(-0.15, 0.15)

ax = axes[1]
ax.hist(x_after, bins=80, color='#4ECDC4', alpha=0.8, edgecolor='white', linewidth=0.5)
from scipy.stats import kurtosis
ax.set_title(f'After Rotation (kurtosis={kurtosis(x_after):.1f})')
ax.set_xlabel('Coordinate Value')
ax.set_ylabel('Count')

plt.tight_layout()
plt.savefig('assets/gaussianization.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ assets/gaussianization.png")


# ============================================================
# 图3: 显存对比（不同分辨率）
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

scenarios = ['480P\n17f', '480P\n49f', '480P\n81f', '720P\n49f', '720P\n81f']
fp32_mb = [91, 238, 384, 548, 886]
bit3_mb = [9, 24, 38, 55, 88]

x = np.arange(len(scenarios))
width = 0.35

bars1 = ax.bar(x - width/2, fp32_mb, width, label='fp32 Cache', color='#FF6B6B', edgecolor='white', linewidth=2)
bars2 = ax.bar(x + width/2, bit3_mb, width, label='3-bit VedioQuant', color='#4ECDC4', edgecolor='white', linewidth=2)

ax.set_xlabel('Video Configuration')
ax.set_ylabel('Cache Memory (MB)')
ax.set_title('Cache Memory: fp32 vs VedioQuant 3-bit')
ax.set_xticks(x)
ax.set_xticklabels(scenarios)
ax.legend()

for bar, val in zip(bars1, fp32_mb):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
            f'{val}', ha='center', va='bottom', fontsize=10)
for bar, val in zip(bars2, bit3_mb):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
            f'{val}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('assets/memory_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ assets/memory_comparison.png")


# ============================================================
# 图4: 30层全缓存的显存对比
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

categories = ['2 Layers', '5 Layers', '10 Layers', '30 Layers']
fp32_gb = [0.886, 2.16, 4.33, 12.98]
bit3_gb = [0.088, 0.221, 0.441, 1.29]

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, fp32_gb, width, label='fp32 Cache', color='#FF6B6B', edgecolor='white', linewidth=2)
bars2 = ax.bar(x + width/2, bit3_gb, width, label='3-bit VedioQuant', color='#4ECDC4', edgecolor='white', linewidth=2)

ax.axhline(y=24-5, color='orange', linestyle='--', alpha=0.7, label='24GB GPU limit (minus model)')
ax.set_xlabel('Cached Layers (720P, 81 frames)')
ax.set_ylabel('Cache Memory (GB)')
ax.set_title('Full-Layer Caching: Enabling Consumer GPUs')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend(loc='upper left')

for bar, val in zip(bars1, fp32_gb):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{val:.1f}GB', ha='center', va='bottom', fontsize=10)
for bar, val in zip(bars2, bit3_gb):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{val:.1f}GB', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('assets/full_cache_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ assets/full_cache_comparison.png")


# ============================================================
# 图5: TurboQuant 算法流程图（简化版，用 matplotlib 画）
# ============================================================
fig, ax = plt.subplots(figsize=(14, 4))
ax.set_xlim(0, 14)
ax.set_ylim(0, 4)
ax.axis('off')

boxes = [
    (0.5, 1.5, 'Input Vector\nx ∈ R^d\n(fp32, 16 bits)', '#FFE66D'),
    (3.0, 1.5, 'Norm Split\nγ = ||x||\nx̂ = x/γ', '#FF6B6B'),
    (5.5, 1.5, 'Random\nRotation\nΠx̂ → Gaussian', '#4ECDC4'),
    (8.0, 1.5, 'Scalar\nQuantize\n(Lloyd-Max)', '#45B7D1'),
    (10.5, 1.5, 'Compressed\nγ + indices\n(b bits/coord)', '#96CEB4'),
]

for x, y, text, color in boxes:
    rect = plt.Rectangle((x-0.9, y-0.8), 1.8, 1.6, facecolor=color,
                          edgecolor='#333', linewidth=2, alpha=0.9, zorder=2)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold', zorder=3)

# 箭头
for i in range(len(boxes)-1):
    x1 = boxes[i][0] + 0.9
    x2 = boxes[i+1][0] - 0.9
    ax.annotate('', xy=(x2, 1.5), xytext=(x1, 1.5),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))

# 压缩比标注
ax.text(7, 3.5, '32 bits → 3 bits = 10.7× compression', ha='center',
        fontsize=14, fontweight='bold', color='#333',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F7DC6F', alpha=0.8))

plt.savefig('assets/algorithm_flow.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ assets/algorithm_flow.png")

print("\n所有图表已生成到 assets/ 目录")
