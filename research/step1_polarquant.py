"""
Step 1: PolarQuant 复现
- 范数分离
- 随机旋转（使坐标高斯化）
- 标量量化（Lloyd-Max）
"""

import numpy as np
from scipy.stats import kurtosis


# ============================================================
# 第一步：生成一个模拟的 KV 向量（模拟真实分布：大部分值集中在0附近，少量离群值）
# ============================================================

np.random.seed(42)
d = 128  # 向量维度

# 模拟真实 KV 向量的分布：混合高斯 + 离群值
# 99% 的坐标来自窄高斯，1% 是离群值
x = np.random.randn(d) * 0.1  # 大部分值在 -0.3~0.3 之间
outlier_idx = np.random.choice(d, size=max(1, d // 100), replace=False)
x[outlier_idx] = np.random.randn(len(outlier_idx)) * 3.0  # 少量极端值

print("=" * 60)
print("原始向量 x")
print("=" * 60)
print(f"维度: {d}")
print(f"均值: {x.mean():.4f}")
print(f"标准差: {x.std():.4f}")
print(f"最小值: {x.min():.4f}, 最大值: {x.max():.4f}")
print(f"峰度(kurtosis): {kurtosis(x):.1f}  (高斯=0, 越大越有离群值)")
print()


# ============================================================
# 第二步：范数分离 — 把"长度"和"方向"分开
# ============================================================

gamma = np.linalg.norm(x)  # 范数（向量长度）
x_hat = x / gamma           # 单位向量（只保留方向）

print("=" * 60)
print("范数分离")
print("=" * 60)
print(f"范数 γ = {gamma:.4f} (单独用 float32 存储，不量化)")
print(f"单位向量 x̂ 的范数 = {np.linalg.norm(x_hat):.4f} (应为1.0)")
print()


# ============================================================
# 第三步：随机旋转 — 让坐标分布变高斯
# ============================================================
# 用随机正交矩阵（QR分解生成）
# 实际工程中用 Hadamard 矩阵更快，这里用 QR 更直观

# 生成随机正交矩阵
random_matrix = np.random.randn(d, d)
Pi, _ = np.linalg.qr(random_matrix)  # QR分解，Pi 是正交矩阵

# 验证正交性：Pi^T @ Pi 应该是单位矩阵
identity_check = Pi.T @ Pi
print("=" * 60)
print("随机旋转矩阵 Π")
print("=" * 60)
print(f"矩阵形状: {Pi.shape}")
print(f"正交性验证 (Π^T·Π 对角线均值): {np.diag(identity_check).mean():.6f} (应为1.0)")
print(f"正交性验证 (Π^T·Π 非对角线均值): {(identity_check - np.eye(d)).mean():.6f} (应为0.0)")
print()

# 旋转
x_rotated = Pi @ x_hat

print("=" * 60)
print("旋转前 vs 旋转后")
print("=" * 60)
print(f"旋转前 x̂ 峰度: {kurtosis(x_hat):.1f}")
print(f"旋转后 Πx̂ 峰度: {kurtosis(x_rotated):.1f}  (越接近0越像高斯)")
print(f"旋转后标准差: {x_rotated.std():.6f}")
print(f"理论值 1/√d:   {1/np.sqrt(d):.6f}")
print(f"比值: {x_rotated.std() / (1/np.sqrt(d)):.3f} (越接近1.0越好)")
print()

# 验证：旋转不改变向量长度（正交变换保范数）
print(f"旋转前范数: {np.linalg.norm(x_hat):.6f}")
print(f"旋转后范数: {np.linalg.norm(x_rotated):.6f}")
print()


# ============================================================
# 第四步：标量量化 — Lloyd-Max 最优量化器
# ============================================================
# 对高斯分布数据，Lloyd-Max 给出误差最小的量化方案

def lloyd_max_quantizer(data, num_levels, max_iter=100):
    """
    Lloyd-Max 最优标量量化器
    输入: data (一堆数), num_levels (量化级数, 比如 2-bit = 4级)
    输出: levels (量化值), boundaries (分界点)
    """
    # 初始化：均匀分布量化级
    min_val, max_val = data.min(), data.max()
    levels = np.linspace(min_val, max_val, num_levels)

    for iteration in range(max_iter):
        # 1. 计算分界点：相邻量化值的中点
        boundaries = (levels[:-1] + levels[1:]) / 2

        # 2. 把每个数据点分配到最近的量化值
        # 加上 -inf 和 +inf 边界
        all_boundaries = np.concatenate([[-np.inf], boundaries, [np.inf]])

        new_levels = np.zeros(num_levels)
        for i in range(num_levels):
            mask = (data >= all_boundaries[i]) & (data < all_boundaries[i + 1])
            if mask.sum() > 0:
                new_levels[i] = data[mask].mean()  # 每个区间的均值作为新量化值
            else:
                new_levels[i] = levels[i]

        # 3. 检查收敛
        if np.allclose(levels, new_levels, atol=1e-8):
            break
        levels = new_levels

    return levels, boundaries


# 2.5-bit PolarQuant: Step 1 用 b-1 = 2.5-1 = 1.5 bits？
# 实际上 bit 数必须是整数级数，这里演示用 2-bit（4级）
bits_step1 = 2  # 对应 3-bit 总配置 (step1=2bit + step2=1bit)
num_levels = 2 ** bits_step1  # 4 个量化级

print("=" * 60)
print(f"Lloyd-Max 量化 ({bits_step1}-bit, {num_levels} 级)")
print("=" * 60)

# 用旋转后的数据训练量化器
levels, boundaries = lloyd_max_quantizer(x_rotated, num_levels)

print(f"量化级 (levels): {np.round(levels, 4)}")
print(f"分界点 (boundaries): {np.round(boundaries, 4)}")
print()

# 对旋转后的向量做量化
def quantize(data, levels, boundaries):
    """把每个值映射到最近的量化级"""
    all_boundaries = np.concatenate([[-np.inf], boundaries, [np.inf]])
    result = np.zeros_like(data)
    indices = np.zeros(len(data), dtype=int)
    for i in range(len(levels)):
        mask = (data >= all_boundaries[i]) & (data < all_boundaries[i + 1])
        result[mask] = levels[i]
        indices[mask] = i
    return result, indices

x_quantized, indices = quantize(x_rotated, levels, boundaries)

# 存储的是 indices（每个只需 bits_step1 个bit）+ levels表（很小，只有num_levels个float）


# ============================================================
# 第五步：解压（反量化 + 反旋转）
# ============================================================

# 反旋转：正交矩阵的逆就是转置
x_hat_reconstructed = Pi.T @ x_quantized

# 乘回范数
x_reconstructed = x_hat_reconstructed * gamma


# ============================================================
# 第六步：评估压缩质量
# ============================================================

print("=" * 60)
print("压缩质量评估")
print("=" * 60)

# 余弦相似度
cos_sim = np.dot(x, x_reconstructed) / (np.linalg.norm(x) * np.linalg.norm(x_reconstructed))

# MSE
mse = np.mean((x - x_reconstructed) ** 2)

# 压缩比
original_bits = 32  # float32
compressed_bits = bits_step1  # 每个坐标只需 bits_step1 个bit（忽略范数和量化表的存储）
compression_ratio = original_bits / compressed_bits

print(f"余弦相似度: {cos_sim:.4f}")
print(f"MSE: {mse:.6f}")
print(f"压缩比: {compression_ratio:.1f}× ({original_bits}bit → {compressed_bits}bit/坐标)")
print()

# 逐坐标对比（前10个）
print("前10个坐标对比:")
print(f"{'原始':>10s} {'重建':>10s} {'误差':>10s}")
for i in range(10):
    print(f"{x[i]:10.4f} {x_reconstructed[i]:10.4f} {x[i]-x_reconstructed[i]:10.4f}")

print()
print("=" * 60)
print("总结: Step 1 PolarQuant 完成")
print("=" * 60)
print(f"  旋转使峰度从 {kurtosis(x_hat):.1f} 降到 {kurtosis(x_rotated):.1f}")
print(f"  {bits_step1}-bit 量化余弦相似度: {cos_sim:.4f}")
print(f"  残差 (供 Step 2 QJL 使用): r = 旋转后真实值 - 量化值")

residual = x_rotated - x_quantized
print(f"  残差范数: {np.linalg.norm(residual):.6f}")
print(f"  残差占原向量比例: {np.linalg.norm(residual)/np.linalg.norm(x_rotated)*100:.1f}%")
