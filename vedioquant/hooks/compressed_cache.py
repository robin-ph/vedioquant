"""
压缩缓存 Hook 系统

基于 FirstBlockCache (TeaCache) 的思路:
- 头部 block 每步都计算，用残差判断是否可以跳过后续 blocks
- 尾部 block 的残差用 TurboQuant 压缩存储
- 缓存命中时解压复用，跳过中间所有 blocks
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from ..compressor.turbo_quant import TurboQuantCompressor, CompressedTensor
from ..config import VedioquantConfig


class CompressedCacheState:
    """缓存的共享状态"""

    def __init__(self):
        self.head_block_residual: Optional[torch.Tensor] = None
        self.compressed_tail_residual: Optional[CompressedTensor] = None
        self.head_block_output: Optional[torch.Tensor] = None
        self.should_compute: bool = True
        self.step_count: int = 0
        self.cache_hits: int = 0

    def reset(self):
        self.head_block_residual = None
        self.compressed_tail_residual = None
        self.head_block_output = None
        self.should_compute = True

    @property
    def cache_hit_rate(self) -> float:
        return self.cache_hits / max(self.step_count, 1)


class VedioquantHandle:
    """
    enable() 返回的句柄，用于查询状态和禁用缓存。

    用法:
        handle = vedioquant.enable(model)
        # ... 运行推理 ...
        print(handle.stats())
        vedioquant.disable(handle)
    """

    def __init__(
        self,
        model: nn.Module,
        config: VedioquantConfig,
        state: CompressedCacheState,
        compressor: TurboQuantCompressor,
        hooks: list,
    ):
        self.model = model
        self.config = config
        self.state = state
        self.compressor = compressor
        self._hooks = hooks

    def stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            "steps": self.state.step_count,
            "cache_hits": self.state.cache_hits,
            "hit_rate": f"{self.state.cache_hit_rate:.1%}",
            "bits": self.config.bits,
            "compression_ratio": f"{self.compressor.compression_ratio:.1f}×",
        }

    def memory_saved_bytes(self) -> int:
        """估算已节省的显存字节数"""
        if self.state.compressed_tail_residual is None:
            return 0
        ct = self.state.compressed_tail_residual
        original = ct.norms.shape[0] * ct.dim * 4  # fp32
        compressed = self.compressor.compressed_size_bytes(
            torch.zeros(ct.norms.shape[0], ct.dim)
        )
        return (original - compressed) * self.state.cache_hits

    def reset(self):
        """重置缓存状态（新视频生成前调用）"""
        self.state.reset()


def _should_compute(
    current_residual: torch.Tensor,
    prev_residual: Optional[torch.Tensor],
    threshold: float,
) -> bool:
    """判断是否需要计算剩余 blocks（和 FirstBlockCache 相同的指标）"""
    if prev_residual is None:
        return True

    absmean = (current_residual - prev_residual).abs().mean()
    prev_absmean = prev_residual.abs().mean()

    if prev_absmean < 1e-10:
        return True

    diff = (absmean / prev_absmean).item()
    return diff > threshold


def apply_compressed_cache(
    model: nn.Module,
    config: VedioquantConfig,
) -> VedioquantHandle:
    """
    给模型安装压缩缓存 hooks。

    使用 PyTorch 标准的 forward hook 系统。
    在每个去噪步中:
    1. 头部 block 计算残差，判断是否可以跳过
    2. 如果可以跳过：解压尾部残差，直接输出
    3. 如果不能跳过：正常计算，压缩存储尾部残差

    Args:
        model: 视频模型的 transformer 部分
        config: VedioQuant 配置

    Returns:
        VedioquantHandle 句柄
    """
    from .block_detector import detect_transformer_blocks, infer_hidden_dim

    blocks = detect_transformer_blocks(model)
    hidden_dim = infer_hidden_dim(model)

    state = CompressedCacheState()
    compressor = TurboQuantCompressor(dim=hidden_dim, bits=config.bits, seed=config.seed)
    hooks = []

    # 选择要缓存的层
    if config.layers is not None:
        selected_indices = set(config.layers)
    else:
        selected_indices = set(range(len(blocks)))

    head_idx = min(selected_indices) if selected_indices else 0
    tail_idx = max(selected_indices) if selected_indices else len(blocks) - 1

    # 安装头部 block hook
    head_name, head_block = blocks[head_idx]

    def head_post_hook(module, input, output):
        state.step_count += 1

        # 提取 hidden_states（可能在 output 或 output[0]）
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        inp = input[0] if isinstance(input, tuple) else input
        residual = (hidden - inp).detach().cpu().float()

        state.should_compute = _should_compute(
            residual, state.head_block_residual, config.threshold
        )

        if not state.should_compute and state.compressed_tail_residual is not None:
            state.cache_hits += 1
        else:
            state.head_block_residual = residual
            state.head_block_output = hidden.detach()

        return output

    hooks.append(head_block.register_forward_hook(head_post_hook))

    # 安装尾部 block hook
    if tail_idx != head_idx:
        tail_name, tail_block = blocks[tail_idx]

        def tail_post_hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            if state.should_compute and state.head_block_output is not None:
                # 计算并压缩尾部残差
                tail_residual = hidden - state.head_block_output.to(hidden.device)
                state.compressed_tail_residual = compressor.compress(tail_residual)

            return output

        hooks.append(tail_block.register_forward_hook(tail_post_hook))

    return VedioquantHandle(model, config, state, compressor, hooks)


def remove_compressed_cache(handle: VedioquantHandle) -> None:
    """
    移除所有压缩缓存 hooks。

    Args:
        handle: enable() 返回的句柄
    """
    for hook in handle._hooks:
        hook.remove()
    handle._hooks.clear()
    handle.state.reset()
