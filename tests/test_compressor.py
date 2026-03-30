"""TurboQuant 压缩器测试"""

import torch
import numpy as np
import pytest
from vedioquant.compressor import PolarQuantCompressor, TurboQuantCompressor


class TestPolarQuant:
    def test_compress_decompress_basic(self):
        """基本压缩解压流程"""
        dim = 128
        comp = PolarQuantCompressor(dim=dim, bits=3)

        x = torch.randn(10, dim)
        norms, indices = comp.compress(x)
        x_recon = comp.decompress(norms, indices)

        assert x_recon.shape == x.shape
        assert norms.shape == (10,)
        assert indices.shape == (10, dim)

    def test_cosine_similarity_3bit(self):
        """3-bit 压缩余弦相似度 > 0.95"""
        dim = 1536  # Wan2.1 hidden_dim
        comp = PolarQuantCompressor(dim=dim, bits=3)

        x = torch.randn(50, dim)
        norms, indices = comp.compress(x)
        x_recon = comp.decompress(norms, indices)

        for i in range(50):
            v = x[i].numpy()
            r = x_recon[i].numpy()
            cos = np.dot(v, r) / (np.linalg.norm(v) * np.linalg.norm(r))
            assert cos > 0.95, f"向量 {i}: 余弦相似度 {cos:.4f} < 0.95"

    def test_cosine_similarity_2bit(self):
        """2-bit 压缩余弦相似度 > 0.85"""
        dim = 1536
        comp = PolarQuantCompressor(dim=dim, bits=2)

        x = torch.randn(50, dim)
        norms, indices = comp.compress(x)
        x_recon = comp.decompress(norms, indices)

        cos_sims = []
        for i in range(50):
            v = x[i].numpy()
            r = x_recon[i].numpy()
            cos_sims.append(np.dot(v, r) / (np.linalg.norm(v) * np.linalg.norm(r)))
        assert np.mean(cos_sims) > 0.85

    def test_zero_vector(self):
        """零向量不崩溃"""
        dim = 128
        comp = PolarQuantCompressor(dim=dim, bits=3)
        x = torch.zeros(1, dim)
        norms, indices = comp.compress(x)
        x_recon = comp.decompress(norms, indices)
        assert torch.allclose(x_recon, x, atol=1e-6)

    def test_single_vector(self):
        """单向量输入"""
        dim = 256
        comp = PolarQuantCompressor(dim=dim, bits=3)
        x = torch.randn(dim)
        norms, indices = comp.compress(x)
        x_recon = comp.decompress(norms, indices)
        assert x_recon.shape == (1, dim)


class TestTurboQuant:
    def test_compress_2d(self):
        """2D 张量压缩"""
        dim = 256
        comp = TurboQuantCompressor(dim=dim, bits=3)
        x = torch.randn(20, dim)
        compressed = comp.compress(x)
        recon = comp.decompress(compressed)
        assert recon.shape == x.shape

    def test_compress_3d(self):
        """3D 张量压缩（典型注意力输出形状）"""
        dim = 1536
        comp = TurboQuantCompressor(dim=dim, bits=3)
        x = torch.randn(1, 64, dim)
        compressed = comp.compress(x)
        recon = comp.decompress(compressed)
        assert recon.shape == x.shape

    def test_compress_5d(self):
        """5D 张量压缩（最后一维作为向量维度）"""
        dim = 64
        comp = TurboQuantCompressor(dim=dim, bits=3)
        x = torch.randn(1, 8, 2, 4, dim)
        compressed = comp.compress(x)
        recon = comp.decompress(compressed)
        assert recon.shape == x.shape

    def test_compression_ratio(self):
        """压缩比计算"""
        comp = TurboQuantCompressor(dim=1536, bits=3)
        assert comp.compression_ratio == 32.0 / 3

    def test_device_transfer(self):
        """解压到指定设备"""
        dim = 128
        comp = TurboQuantCompressor(dim=dim, bits=3)
        x = torch.randn(5, dim)
        compressed = comp.compress(x)
        recon = comp.decompress(compressed, device="cpu", dtype=torch.float32)
        assert recon.device.type == "cpu"
        assert recon.dtype == torch.float32
