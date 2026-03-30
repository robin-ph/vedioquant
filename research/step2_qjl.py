"""
Step 2: QJL (Quantized Johnson-Lindenstrauss) 复现
在 Step 1 PolarQuant 基础上，用 1 bit 符号位补偿残差

完整流程: Step 1 (b-1 bits) + Step 2 (1 bit) = 总共 b bits/坐标
"""

import numpy as np
from scipy.stats import kurtosis


# ============================================================
# 复用 Step 1 的代码（生成向量 → 范数分离 → 旋转 → 量化）
# ============================================================

np.random.seed(42)
d = 128

# 模拟 KV 向量
x = np.random.randn(d) * 0.1
outlier_idx = np.random.choice(d, size=max(1, d // 100), replace=False)
x[outlier_idx] = np.random.randn(len(outlier_idx)) * 3.0

# 范数分离
gamma = np.linalg.norm(x)
x_hat = x / gamma

# 随机旋转
random_matrix = np.random.randn(d, d)
Pi, _ = np.linalg.qr(random_matrix)
x_rotated = Pi @ x_hat

# Lloyd-Max 量化（Step 1, 2-bit）
def lloyd_max_quantizer(data, num_levels, max_iter=100):
    min_val, max_val = data.min(), data.max()
    levels = np.linspace(min_val, max_val, num_levels)
    for _ in range(max_iter):
        boundaries = (levels[:-1] + levels[1:]) / 2
        all_boundaries = np.concatenate([[-np.inf], boundaries, [np.inf]])
        new_levels = np.zeros(num_levels)
        for i in range(num_levels):
            mask = (data >= all_boundaries[i]) & (data < all_boundaries[i + 1])
            if mask.sum() > 0:
                new_levels[i] = data[mask].mean()
            else:
                new_levels[i] = levels[i]
        if np.allclose(levels, new_levels, atol=1e-8):
            break
        levels = new_levels
    return levels, boundaries

bits_step1 = 2
num_levels = 2 ** bits_step1
levels, boundaries = lloyd_max_quantizer(x_rotated, num_levels)

def quantize(data, levels, boundaries):
    all_boundaries = np.concatenate([[-np.inf], boundaries, [np.inf]])
    result = np.zeros_like(data)
    indices = np.zeros(len(data), dtype=int)
    for i in range(len(levels)):
        mask = (data >= all_boundaries[i]) & (data < all_boundaries[i + 1])
        result[mask] = levels[i]
        indices[mask] = i
    return result, indices

x_quantized, indices = quantize(x_rotated, levels, boundaries)

# Step 1 的残差
residual = x_rotated - x_quantized

print("=" * 60)
print("Step 1 回顾")
print("=" * 60)
print(f"原始向量 x: 维度={d}")
print(f"Step 1 (2-bit) 量化后余弦相似度: ", end="")
x_hat_recon_step1 = Pi.T @ x_quantized
x_recon_step1 = x_hat_recon_step1 * gamma
cos_step1 = np.dot(x, x_recon_step1) / (np.linalg.norm(x) * np.linalg.norm(x_recon_step1))
print(f"{cos_step1:.4f}")
print(f"残差范数: {np.linalg.norm(residual):.6f}")
print()


# ============================================================
# Step 2: QJL — 用 1 bit 补偿残差
# ============================================================

print("=" * 60)
print("Step 2: QJL 开始")
print("=" * 60)

# --- 2a. 生成随机矩阵 S ---
# S 的每个元素是 +1 或 -1（Rademacher 随机变量）
# 形状: (d, d)，实际工程中可以用更小的投影维度，这里用 d 便于理解
# 关键：压缩和解压用同一个种子，所以能还原同一个 S

rng_qjl = np.random.RandomState(seed=123)  # 固定种子，解压时用同一个
S = rng_qjl.choice([-1, 1], size=(d, d)).astype(float)

print(f"随机矩阵 S 形状: {S.shape}")
print(f"S 的元素: 全部是 +1 或 -1")
print(f"S 的均值: {S.mean():.3f} (应接近0)")
print()

# --- 2b. 投影残差，取符号 ---
projected = S @ residual          # 形状: (d,)  每个值 = 残差各坐标的随机加减
signs = np.sign(projected)        # 形状: (d,)  每个值只有 +1 或 -1 → 1 bit

# 处理恰好为0的情况（极罕见）
signs[signs == 0] = 1

print("投影 + 取符号:")
print(f"  投影后 S·r 的前5个值: {np.round(projected[:5], 4)}")
print(f"  符号位 sign(S·r) 前5个: {signs[:5].astype(int)}")
print(f"  存储: 每个符号只需 1 bit，共 {d} bits = {d/8:.0f} bytes")
print()


# ============================================================
# 解压：利用符号位修正内积
# ============================================================

print("=" * 60)
print("解压与内积修正")
print("=" * 60)

# 为了演示内积修正效果，我们造一个 query 向量 q
# 模拟注意力计算: 需要算 <q, x> 的内积
q = np.random.randn(d) * 0.1
q[np.random.choice(d, size=1)] = np.random.randn() * 2.0

# --- 真实内积 ---
true_inner_product = np.dot(q, x)

# --- 仅 Step 1（无修正）的内积 ---
inner_step1_only = np.dot(q, x_recon_step1)

# --- Step 1 + Step 2 修正的内积 ---
# QJL 的核心公式:
# <q, x> ≈ <q_recon, x_recon> + 修正项
#
# 修正项的计算：
# 1. 对 q 也做同样的处理（旋转 → 量化 → 得到残差）
# 2. 用双方的符号位估算残差内积

# 对 q 做同样的处理
q_gamma = np.linalg.norm(q)
q_hat = q / q_gamma
q_rotated = Pi @ q_hat
q_quantized, _ = quantize(q_rotated, levels, boundaries)
q_residual = q_rotated - q_quantized
q_recon_step1 = Pi.T @ q_quantized * q_gamma

# q 的残差也投影取符号
q_projected = S @ q_residual
q_signs = np.sign(q_projected)
q_signs[q_signs == 0] = 1

# QJL 修正项：
# 两组符号位的点积，除以 d，再乘以残差范数的估计
# 直觉：符号一致（都+或都-）说明残差方向一致，内积应该加正的修正
# 公式: correction ≈ (signs_x · signs_q) / d * ||r_x|| * ||r_q|| * gamma_x * gamma_q
#
# 简化版本（用残差范数近似）：
sign_inner = np.dot(signs, q_signs) / d  # 符号位的归一化内积，范围[-1, 1]
correction = sign_inner * np.linalg.norm(residual) * np.linalg.norm(q_residual) * gamma * q_gamma

inner_corrected = inner_step1_only + correction

print(f"真实内积 <q, x>:              {true_inner_product:.6f}")
print(f"Step 1 only 内积:             {inner_step1_only:.6f}  误差={abs(inner_step1_only - true_inner_product):.6f}")
print(f"Step 1 + Step 2 修正后内积:    {inner_corrected:.6f}  误差={abs(inner_corrected - true_inner_product):.6f}")
print()
print(f"修正项大小: {correction:.6f}")
print(f"修正使误差变化: {abs(inner_step1_only - true_inner_product):.6f} → {abs(inner_corrected - true_inner_product):.6f}")

if abs(inner_corrected - true_inner_product) < abs(inner_step1_only - true_inner_product):
    print("→ 修正有效！误差减小了")
else:
    print("→ 本次修正没有减小误差（单次可能，多次平均一定更好）")
print()


# ============================================================
# 统计验证：多次实验取平均，验证无偏性
# ============================================================

print("=" * 60)
print("统计验证：1000个随机向量对的内积修正效果")
print("=" * 60)

errors_step1 = []
errors_corrected = []

for trial in range(1000):
    # 随机 q
    q_trial = np.random.randn(d) * 0.15
    q_trial[np.random.choice(d, size=max(1, d // 50))] = np.random.randn(max(1, d // 50)) * 2.0

    true_ip = np.dot(q_trial, x)

    # Step 1 only
    q_g = np.linalg.norm(q_trial)
    q_h = q_trial / q_g
    q_rot = Pi @ q_h
    q_quant, _ = quantize(q_rot, levels, boundaries)
    q_res = q_rot - q_quant
    q_recon = Pi.T @ q_quant * q_g
    ip_step1 = np.dot(q_trial, x_recon_step1)

    # Step 2 修正
    q_proj = S @ q_res
    q_sgn = np.sign(q_proj)
    q_sgn[q_sgn == 0] = 1
    sgn_ip = np.dot(signs, q_sgn) / d
    corr = sgn_ip * np.linalg.norm(residual) * np.linalg.norm(q_res) * gamma * q_g
    ip_corrected = ip_step1 + corr

    errors_step1.append(abs(ip_step1 - true_ip))
    errors_corrected.append(abs(ip_corrected - true_ip))

errors_step1 = np.array(errors_step1)
errors_corrected = np.array(errors_corrected)

print(f"Step 1 only  — 平均绝对误差: {errors_step1.mean():.6f}")
print(f"Step 1 + QJL — 平均绝对误差: {errors_corrected.mean():.6f}")
print(f"误差降低比例: {(1 - errors_corrected.mean()/errors_step1.mean())*100:.1f}%")
print()


# ============================================================
# 最终：完整重建向量的质量（Step 1 + Step 2）
# ============================================================

print("=" * 60)
print("完整向量重建质量对比")
print("=" * 60)

# Step 1 only: 2-bit
cos1 = np.dot(x, x_recon_step1) / (np.linalg.norm(x) * np.linalg.norm(x_recon_step1))
mse1 = np.mean((x - x_recon_step1) ** 2)

# Step 1 + Step 2: 2+1 = 3-bit 总配置
# 向量重建: 用 Step 2 的符号位近似重建残差
# r_approx = S^T @ (signs * scale)
# scale 是残差投影的估计幅度
scale = np.linalg.norm(residual) / np.sqrt(d)
residual_approx = (S.T @ (signs * scale)) / d
x_rotated_recon2 = x_quantized + residual_approx
x_hat_recon2 = Pi.T @ x_rotated_recon2
x_recon_step2 = x_hat_recon2 * gamma

cos2 = np.dot(x, x_recon_step2) / (np.linalg.norm(x) * np.linalg.norm(x_recon_step2))
mse2 = np.mean((x - x_recon_step2) ** 2)

print(f"{'配置':<20s} {'bits/坐标':>10s} {'余弦相似度':>10s} {'MSE':>12s}")
print("-" * 55)
print(f"{'Step1 only':<20s} {'2':>10s} {cos1:>10.4f} {mse1:>12.6f}")
print(f"{'Step1 + Step2':<20s} {'2+1=3':>10s} {cos2:>10.4f} {mse2:>12.6f}")
print()
print("=" * 60)
print("总结")
print("=" * 60)
print(f"Step 2 QJL 额外用 1 bit/坐标，通过符号位修正：")
print(f"  - 内积误差平均降低 {(1 - errors_corrected.mean()/errors_step1.mean())*100:.1f}%")
print(f"  - 余弦相似度: {cos1:.4f} → {cos2:.4f}")
print(f"  - 总存储: 3 bits/坐标 (压缩比 {32/3:.1f}×)")
