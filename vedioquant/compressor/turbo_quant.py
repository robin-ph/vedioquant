"""
TurboQuant: PolarQuant + QJL 组合压缩器

对外的主要压缩接口。
"""

import torch
from typing import NamedTuple, Optional
from .polar_quant import PolarQuantCompressor


class CompressedTensor(NamedTuple):
    """压缩后的张量"""
    norms: torch.Tensor      # (N,) 向量范数
    indices: torch.Tensor     # (N, d) 量化索引
    original_shape: tuple     # 原始张量形状
    dim: int                  # 向量维度


class TurboQuantCompressor:
    """
    TurboQuant 压缩器

    用法:
        compressor = TurboQuantCompressor(dim=1536, bits=3)
        compressed = compressor.compress(tensor)
        reconstructed = compressor.decompress(compressed)

    Args:
        dim: 向量维度（模型 hidden_dim）
        bits: 量化位数 (2, 3, 4)
        seed: 随机种子
    """

    def __init__(self, dim: int, bits: int = 3, seed: int = 42):
        self.dim = dim
        self.bits = bits
        self.polar = PolarQuantCompressor(dim=dim, bits=bits, seed=seed)

    def compress(self, tensor: torch.Tensor) -> CompressedTensor:
        """
        压缩任意形状的张量（最后一维作为向量维度）。

        Args:
            tensor: (..., dim) 的张量

        Returns:
            CompressedTensor
        """
        original_shape = tensor.shape
        flat = tensor.reshape(-1, tensor.shape[-1])  # (N, d)

        norms, indices = self.polar.compress(flat)

        return CompressedTensor(
            norms=norms,
            indices=indices,
            original_shape=original_shape,
            dim=self.dim,
        )

    def decompress(
        self,
        compressed: CompressedTensor,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        解压回原始形状的张量。

        Args:
            compressed: CompressedTensor
            device: 目标设备
            dtype: 目标精度

        Returns:
            重建的张量，形状与原始一致
        """
        recon = self.polar.decompress(compressed.norms, compressed.indices)
        recon = recon.reshape(compressed.original_shape)
        return recon.to(device=device, dtype=dtype)

    def compressed_size_bytes(self, tensor: torch.Tensor) -> int:
        """计算压缩后字节数"""
        N = tensor.reshape(-1, tensor.shape[-1]).shape[0]
        return self.polar.compressed_size_bytes(N)

    def original_size_bytes(self, tensor: torch.Tensor) -> int:
        """原始字节数"""
        return tensor.nelement() * tensor.element_size()

    @property
    def compression_ratio(self) -> float:
        return self.polar.compression_ratio
