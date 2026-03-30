"""
预计算的 Lloyd-Max 量化码本（标准高斯分布）

旋转后向量坐标服从 N(0, 1/sqrt(d))，是标准高斯的缩放版。
预计算标准高斯的最优码本，使用时只需乘以 scale = 1/sqrt(d)。
避免了每个向量都跑 Lloyd-Max 迭代，大幅加速。
"""

import torch
import numpy as np

# 标准高斯 N(0,1) 的 Lloyd-Max 最优量化码本
# 这些值来自对 N(0,1) 跑 Lloyd-Max 收敛后的结果
_GAUSSIAN_CODEBOOKS = {
    2: {  # 4 levels
        "levels": np.array([-1.5104, -0.4528, 0.4528, 1.5104]),
        "boundaries": np.array([-0.9816, 0.0, 0.9816]),
    },
    3: {  # 8 levels
        "levels": np.array([-2.1520, -1.3440, -0.7560, -0.2451,
                             0.2451, 0.7560, 1.3440, 2.1520]),
        "boundaries": np.array([-1.7480, -1.0500, -0.5006, 0.0,
                                  0.5006, 1.0500, 1.7480]),
    },
    4: {  # 16 levels
        "levels": np.array([-2.7326, -2.0690, -1.6180, -1.2562,
                            -0.9423, -0.6568, -0.3881, -0.1284,
                             0.1284, 0.3881, 0.6568, 0.9423,
                             1.2562, 1.6180, 2.0690, 2.7326]),
        "boundaries": np.array([-2.4008, -1.8435, -1.4371, -1.0993,
                                 -0.7996, -0.5224, -0.2582, 0.0,
                                  0.2582, 0.5224, 0.7996, 1.0993,
                                  1.4371, 1.8435, 2.4008]),
    },
}


def get_codebook(bits: int, scale: float = 1.0):
    """
    获取预计算码本。

    Args:
        bits: 量化位数 (2, 3, 4)
        scale: 缩放因子，对 N(0, sigma^2) 分布使用 sigma 作为 scale

    Returns:
        levels: 量化级 (torch.Tensor)
        boundaries: 分界点 (torch.Tensor)
    """
    if bits not in _GAUSSIAN_CODEBOOKS:
        raise ValueError(f"不支持 {bits}-bit，可选: {list(_GAUSSIAN_CODEBOOKS.keys())}")

    cb = _GAUSSIAN_CODEBOOKS[bits]
    levels = torch.tensor(cb["levels"], dtype=torch.float32) * scale
    boundaries = torch.tensor(cb["boundaries"], dtype=torch.float32) * scale
    return levels, boundaries


def compute_codebook(bits: int, data_std: float = 1.0, max_iter: int = 100):
    """
    用 Lloyd-Max 算法在线计算码本（当预计算不可用时的 fallback）。
    """
    num_levels = 2 ** bits
    # 用标准高斯采样训练
    rng = np.random.RandomState(0)
    samples = rng.randn(100000) * data_std

    levels = np.linspace(samples.min(), samples.max(), num_levels)
    for _ in range(max_iter):
        boundaries = (levels[:-1] + levels[1:]) / 2
        all_bounds = np.concatenate([[-np.inf], boundaries, [np.inf]])
        new_levels = np.zeros(num_levels)
        for i in range(num_levels):
            mask = (samples >= all_bounds[i]) & (samples < all_bounds[i + 1])
            new_levels[i] = samples[mask].mean() if mask.sum() > 0 else levels[i]
        if np.allclose(levels, new_levels, atol=1e-8):
            break
        levels = new_levels

    boundaries = (levels[:-1] + levels[1:]) / 2
    return (
        torch.tensor(levels, dtype=torch.float32),
        torch.tensor(boundaries, dtype=torch.float32),
    )
