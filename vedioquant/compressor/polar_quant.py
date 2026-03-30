"""
PolarQuant: 随机旋转 + 标量量化

GPU-native PyTorch 实现，支持批量压缩。
"""

import torch
import numpy as np
from typing import Tuple, Optional
from .codebook import get_codebook


class PolarQuantCompressor:
    """
    PolarQuant 压缩器

    流程: 范数分离 → 随机正交旋转 → 标量量化（预计算码本）

    Args:
        dim: 向量维度
        bits: 量化位数 (2, 3, 4)
        seed: 随机种子（生成正交矩阵用）
    """

    def __init__(self, dim: int, bits: int = 3, seed: int = 42):
        self.dim = dim
        self.bits = bits
        self.num_levels = 2 ** bits
        self.seed = seed

        # 生成随机正交矩阵
        rng = np.random.RandomState(seed)
        R = rng.randn(dim, dim).astype(np.float32)
        Q, _ = np.linalg.qr(R)
        self.Pi = torch.tensor(Q, dtype=torch.float32)        # (d, d)
        self.Pi_T = self.Pi.T.contiguous()                     # (d, d)

        # 预计算码本（标准高斯缩放到 1/sqrt(d)）
        scale = 1.0 / np.sqrt(dim)
        self.levels, self.boundaries = get_codebook(bits, scale=scale)

    def compress(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        压缩一批向量。

        Args:
            x: (N, d) 或 (d,) 的 float 张量

        Returns:
            norms: (N,) 范数
            indices: (N, d) 量化索引 (uint8)
            这两个加上 self.levels 就能完整还原
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = x.float().cpu()
        N, d = x.shape
        assert d == self.dim, f"维度不匹配: 期望 {self.dim}, 实际 {d}"

        # 范数分离
        norms = torch.norm(x, dim=1)                          # (N,)
        safe_norms = norms.clamp(min=1e-10)
        x_hat = x / safe_norms.unsqueeze(1)                   # (N, d)

        # 旋转: (N, d) @ (d, d)^T = (N, d)
        x_rotated = x_hat @ self.Pi.T                         # (N, d)

        # 量化: 用 bucketize 快速查找最近的量化级
        indices = torch.bucketize(x_rotated, self.boundaries)  # (N, d), int64
        indices = indices.to(torch.uint8)

        return norms, indices

    def decompress(self, norms: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        解压一批向量。

        Args:
            norms: (N,) 范数
            indices: (N, d) 量化索引

        Returns:
            x_recon: (N, d) 重建的向量
        """
        # 反量化
        x_quantized = self.levels[indices.long()]              # (N, d)

        # 反旋转
        x_hat_recon = x_quantized @ self.Pi                   # (N, d)

        # 乘回范数
        x_recon = x_hat_recon * norms.unsqueeze(1)

        return x_recon

    def compressed_size_bytes(self, N: int) -> int:
        """计算 N 个向量压缩后的字节数"""
        norms_bytes = N * 2        # float16 存范数
        indices_bytes = N * self.dim * self.bits // 8  # bit-packed indices
        codebook_bytes = self.num_levels * 4           # float32 码本（共享）
        return norms_bytes + indices_bytes + codebook_bytes

    def original_size_bytes(self, N: int) -> int:
        """N 个向量原始 fp32 的字节数"""
        return N * self.dim * 4

    @property
    def compression_ratio(self) -> float:
        """理论压缩比"""
        return 32.0 / self.bits  # fp32 → b-bit
