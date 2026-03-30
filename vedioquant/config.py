"""
VedioQuant 配置
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class VedioquantConfig:
    """
    VedioQuant 配置

    Args:
        bits: 量化位数 (2, 3, 4)，默认 3（甜点配置：压缩比10×，余弦相似度0.98）
        threshold: 缓存复用阈值，越大命中率越高但质量越低，默认 0.05
        layers: 要缓存的层索引，None 表示所有层
        cache_device: 压缩缓存存储位置，"cpu" 节省显存，"same" 留在原设备
        seed: 随机种子（用于生成正交旋转矩阵）
    """
    bits: int = 3
    threshold: float = 0.05
    layers: Optional[List[int]] = None
    cache_device: str = "cpu"
    seed: int = 42
