"""
显存估算工具

不需要运行推理，根据模型配置和视频参数估算显存节省。
"""

from typing import Dict, Optional


def estimate_savings(
    hidden_dim: int = 1536,
    num_layers: int = 30,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    bits: int = 3,
    cache_layers: int = 2,
    vae_spatial: int = 8,
    vae_temporal: int = 4,
    patch_size: tuple = (1, 2, 2),
) -> Dict:
    """
    估算缓存显存节省。

    Args:
        hidden_dim: 模型隐层维度
        num_layers: transformer 层数
        height, width: 视频分辨率
        num_frames: 视频帧数
        bits: TurboQuant 量化位数
        cache_layers: 缓存的层数（FirstBlockCache 默认缓存 2 层残差）
        vae_spatial: VAE 空间压缩倍数
        vae_temporal: VAE 时间压缩倍数
        patch_size: transformer patch 大小

    Returns:
        {
            "tokens": int,
            "fp32_cache_mb": float,
            "compressed_cache_mb": float,
            "saved_mb": float,
            "compression_ratio": float,
            "fits_24gb": {"fp32": bool, "compressed": bool},
        }
    """
    # 计算 token 数
    lat_t = (num_frames - 1) // vae_temporal + 1
    lat_h = height // vae_spatial
    lat_w = width // vae_spatial

    pt, ph, pw = patch_size
    tokens = (lat_t // pt) * (lat_h // ph) * (lat_w // pw)

    # fp32 缓存大小
    per_layer_fp32 = tokens * hidden_dim * 4  # bytes
    total_fp32 = per_layer_fp32 * cache_layers

    # 压缩缓存大小
    # 每个向量: norms(2B float16) + indices(d * bits/8) + codebook(2^bits * 4B, 共享)
    per_vec_compressed = 2 + hidden_dim * bits / 8
    codebook_bytes = (2 ** bits) * 4  # 共享码本
    per_layer_compressed = tokens * per_vec_compressed + codebook_bytes
    total_compressed = per_layer_compressed * cache_layers

    saved = total_fp32 - total_compressed
    ratio = total_fp32 / total_compressed if total_compressed > 0 else float("inf")

    # 假设模型本身占 5GB (1.3B fp32) 或 ~28GB (14B fp32)
    model_size = hidden_dim * num_layers * hidden_dim * 4 * 0.5  # 粗略估算

    def _fmt(b):
        if b > 1024**3:
            return f"{b/1024**3:.2f} GB"
        return f"{b/1024**2:.0f} MB"

    return {
        "tokens": tokens,
        "fp32_cache_bytes": total_fp32,
        "compressed_cache_bytes": int(total_compressed),
        "saved_bytes": int(saved),
        "compression_ratio": round(ratio, 1),
        "fp32_cache": _fmt(total_fp32),
        "compressed_cache": _fmt(total_compressed),
        "saved": _fmt(saved),
        "fits_24gb": {
            "fp32": (total_fp32 + model_size) < 24 * 1024**3,
            "compressed": (total_compressed + model_size) < 24 * 1024**3,
        },
    }
