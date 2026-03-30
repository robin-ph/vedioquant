"""
自动检测 Transformer Block 结构

支持多种视频模型架构:
- Wan2.1: model.blocks (WanTransformerBlock)
- CogVideoX: model.transformer_blocks (CogVideoXBlock)
- HunyuanVideo: model.transformer_blocks
- 通用: 任何包含注意力层的 ModuleList
"""

import torch.nn as nn
from typing import List, Tuple

# diffusers 中常见的 transformer block 容器名
_BLOCK_IDENTIFIERS = (
    "transformer_blocks",
    "single_transformer_blocks",
    "temporal_transformer_blocks",
    "layers",
    "blocks",
)


def detect_transformer_blocks(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """
    自动检测模型中的 transformer block 列表。

    Args:
        model: 视频模型的 transformer 部分

    Returns:
        [(block_name, block_module), ...] 按顺序排列

    Raises:
        ValueError: 如果找不到 transformer blocks
    """
    # 策略 1: 查找已知名称的 ModuleList
    for name, child in model.named_children():
        if name in _BLOCK_IDENTIFIERS and isinstance(child, nn.ModuleList):
            blocks = [(f"{name}.{i}", block) for i, block in enumerate(child)]
            if len(blocks) > 0:
                return blocks

    # 策略 2: 查找类型名包含 "Block" 的 ModuleList
    for name, child in model.named_children():
        if isinstance(child, nn.ModuleList) and len(child) > 0:
            block_type = type(child[0]).__name__
            if "Block" in block_type or "Layer" in block_type:
                return [(f"{name}.{i}", block) for i, block in enumerate(child)]

    # 策略 3: 递归查找包含 attn 模块的子模块
    blocks = []
    for name, module in model.named_modules():
        if hasattr(module, "attn1") or hasattr(module, "self_attn"):
            if name.count(".") <= 2:  # 不要太深层的
                blocks.append((name, module))

    if blocks:
        return blocks

    raise ValueError(
        "无法自动检测 transformer blocks。"
        "请确认模型包含 'blocks', 'transformer_blocks', 或 'layers' 属性。"
        f"模型的子模块: {[n for n, _ in model.named_children()]}"
    )


def infer_hidden_dim(model: nn.Module) -> int:
    """
    推断模型的 hidden_dim。

    从模型 config 或第一个 Linear 层推断。
    """
    # 策略 1: 从 config 读取
    if hasattr(model, "config"):
        cfg = model.config
        if hasattr(cfg, "num_attention_heads") and hasattr(cfg, "attention_head_dim"):
            return cfg.num_attention_heads * cfg.attention_head_dim
        if hasattr(cfg, "hidden_size"):
            return cfg.hidden_size

    # 策略 2: 从第一个注意力层的 to_q 推断
    for name, module in model.named_modules():
        if "to_q" in name and isinstance(module, nn.Linear):
            return module.out_features

    # 策略 3: 从 patch_embedding 推断
    for name, module in model.named_modules():
        if "patch" in name and isinstance(module, (nn.Linear, nn.Conv3d)):
            return module.out_channels if hasattr(module, "out_channels") else module.out_features

    raise ValueError("无法推断 hidden_dim，请在 config 中手动指定")
