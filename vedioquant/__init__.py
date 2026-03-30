"""
VedioQuant - 视频扩散模型推理缓存压缩加速库

将 TurboQuant (KV Cache 极端压缩) 融合进 TeaCache (视频模型缓存加速)，
用 3-bit 量化压缩缓存特征，显存减少 ~10×，质量损失 <2%。

快速使用:
    import vedioquant

    # 一行代码启用压缩缓存
    handle = vedioquant.enable(pipe.transformer, bits=3)

    # 正常运行推理
    output = pipe("a cat sitting on a sofa", num_frames=81)

    # 查看缓存统计
    print(handle.stats())

    # 禁用
    vedioquant.disable(handle)

估算显存:
    savings = vedioquant.estimate_savings(height=720, width=1280, num_frames=81)
    print(savings)
"""

__version__ = "0.1.0"

import torch.nn as nn
from typing import Optional, Dict

from .config import VedioquantConfig
from .hooks.compressed_cache import (
    apply_compressed_cache,
    remove_compressed_cache,
    VedioquantHandle,
)
from .diagnostics.memory import estimate_savings
from .diagnostics.quality import measure_quality


def enable(
    model: nn.Module,
    bits: int = 3,
    threshold: float = 0.05,
    layers: Optional[list] = None,
    seed: int = 42,
) -> VedioquantHandle:
    """
    一行代码启用压缩缓存。

    Args:
        model: 视频模型的 transformer 部分 (如 pipe.transformer)
        bits: 量化位数 (2=激进, 3=推荐, 4=保守)
        threshold: 缓存命中阈值 (越大命中率越高，质量越低)
        layers: 要缓存的层索引，None=全部
        seed: 随机种子

    Returns:
        VedioquantHandle 句柄

    示例:
        handle = vedioquant.enable(pipe.transformer, bits=3)
    """
    config = VedioquantConfig(
        bits=bits,
        threshold=threshold,
        layers=layers,
        seed=seed,
    )
    return apply_compressed_cache(model, config)


def disable(handle: VedioquantHandle) -> None:
    """
    禁用压缩缓存，移除所有 hooks。

    Args:
        handle: enable() 返回的句柄
    """
    remove_compressed_cache(handle)


def diagnose(
    model: nn.Module,
    sample_inputs: Dict,
    bits: int = 3,
) -> Dict:
    """
    诊断模型是否适合 VedioQuant 压缩。

    分析特征分布、测量压缩质量、估算显存节省。

    Args:
        model: transformer 模型
        sample_inputs: forward() 的输入 kwargs
        bits: 量化位数

    Returns:
        诊断报告 dict
    """
    return measure_quality(model, sample_inputs, bits=bits)
