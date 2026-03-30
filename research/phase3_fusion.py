"""
第三阶段：FirstBlockCache (TeaCache) + TurboQuant 融合实验

对比三种配置：
1. baseline: 无缓存，每步全量计算
2. fbc_only: FirstBlockCache 标准缓存（fp32 存储）
3. fbc_turbo: FirstBlockCache + TurboQuant 压缩缓存

在模拟去噪循环中测量：速度、显存、质量
"""

import torch
import numpy as np
from scipy.stats import kurtosis
import time
import sys
import gc

# ============================================================
# PolarQuant 压缩器（从 Step 1/2 复用，优化为批量版本）
# ============================================================

class PolarQuantCompressor:
    """TurboQuant Step 1: PolarQuant 压缩器"""

    def __init__(self, dim, bits=3):
        self.dim = dim
        self.bits = bits
        self.num_levels = 2 ** bits

        # 生成随机正交矩阵（固定种子保证可复现）
        rng = np.random.RandomState(42)
        random_matrix = rng.randn(dim, dim)
        self.Pi, _ = np.linalg.qr(random_matrix)
        self.Pi_T = self.Pi.T  # 预计算转置

    def compress(self, vec_np):
        """压缩单个向量"""
        gamma = np.linalg.norm(vec_np)
        if gamma < 1e-10:
            return {"gamma": 0.0, "indices": np.zeros(self.dim, dtype=np.uint8),
                    "levels": np.zeros(self.num_levels)}

        x_hat = vec_np / gamma
        x_rotated = self.Pi @ x_hat

        # Lloyd-Max 量化
        levels, boundaries = self._lloyd_max(x_rotated, self.num_levels)
        indices = self._quantize_indices(x_rotated, levels, boundaries)

        return {"gamma": gamma, "indices": indices, "levels": levels}

    def decompress(self, compressed):
        """解压单个向量"""
        if compressed["gamma"] < 1e-10:
            return np.zeros(self.dim)

        # 从 indices 重建量化值
        x_quantized = compressed["levels"][compressed["indices"]]
        # 反旋转 + 乘回范数
        return self.Pi_T @ x_quantized * compressed["gamma"]

    def compress_tensor(self, tensor):
        """压缩一个 2D tensor (num_tokens, dim)"""
        arr = tensor.detach().cpu().float().numpy()
        compressed_list = []
        for i in range(arr.shape[0]):
            compressed_list.append(self.compress(arr[i]))
        return compressed_list

    def decompress_tensor(self, compressed_list, device="cpu", dtype=torch.float32):
        """解压回 tensor"""
        vecs = [self.decompress(c) for c in compressed_list]
        return torch.tensor(np.stack(vecs), dtype=dtype, device=device)

    def compressed_size_bytes(self, compressed_list):
        """计算压缩后的字节数"""
        total = 0
        for c in compressed_list:
            total += 4  # gamma (float32)
            total += len(c["indices"]) * (self.bits / 8)  # indices
            total += len(c["levels"]) * 4  # levels (float32)
        return total

    @staticmethod
    def _lloyd_max(data, num_levels, max_iter=50):
        min_val, max_val = data.min(), data.max()
        levels = np.linspace(min_val, max_val, num_levels)
        for _ in range(max_iter):
            boundaries = (levels[:-1] + levels[1:]) / 2
            all_boundaries = np.concatenate([[-np.inf], boundaries, [np.inf]])
            new_levels = np.zeros(num_levels)
            for i in range(num_levels):
                mask = (data >= all_boundaries[i]) & (data < all_boundaries[i + 1])
                new_levels[i] = data[mask].mean() if mask.sum() > 0 else levels[i]
            if np.allclose(levels, new_levels, atol=1e-8):
                break
            levels = new_levels
        return levels, boundaries

    @staticmethod
    def _quantize_indices(data, levels, boundaries):
        all_boundaries = np.concatenate([[-np.inf], boundaries, [np.inf]])
        indices = np.zeros(len(data), dtype=np.uint8)
        for i in range(len(levels)):
            mask = (data >= all_boundaries[i]) & (data < all_boundaries[i + 1])
            indices[mask] = i
        return indices


# ============================================================
# 加载模型
# ============================================================

print("=" * 60)
print("第三阶段：FirstBlockCache + TurboQuant 融合实验")
print("=" * 60)
print()
print("加载 Wan2.1 Transformer...")

from diffusers.models import WanTransformer3DModel

transformer = WanTransformer3DModel.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    subfolder="transformer",
    torch_dtype=torch.float32,
)
transformer = transformer.to("mps")
transformer.eval()
print("模型加载完成")
print()

# 模型参数
config = transformer.config
in_channels = config.in_channels      # 16
num_heads = config.num_attention_heads # 12
head_dim = config.attention_head_dim   # 128
hidden_dim = num_heads * head_dim      # 1536
num_layers = config.num_layers         # 30

print(f"模型: {num_layers} 层, hidden_dim={hidden_dim}, heads={num_heads}×{head_dim}")
print()


# ============================================================
# 构造模拟去噪输入
# ============================================================

def make_inputs(timestep_val, noise_level=0.01, base_hidden=None):
    """构造一步去噪的输入，模拟相邻步间的微小变化"""
    batch = 1
    frames = 1
    height = 16
    width = 16

    if base_hidden is None:
        hidden = torch.randn(batch, in_channels, frames, height, width,
                              dtype=torch.float32, device="mps")
    else:
        # 相邻步：在上一步基础上加微小噪声（模拟去噪过程）
        hidden = base_hidden + torch.randn_like(base_hidden) * noise_level

    encoder_hidden = torch.randn(batch, 32, 4096, dtype=torch.float32, device="mps")
    timestep = torch.tensor([timestep_val], dtype=torch.float32, device="mps")

    return hidden, encoder_hidden, timestep


# ============================================================
# 实验 1: Baseline（无缓存）
# ============================================================

print("=" * 60)
print("实验 1: Baseline（每步全量计算）")
print("=" * 60)

num_steps = 10
base_hidden = None
baseline_outputs = []
baseline_times = []

for step in range(num_steps):
    t = 1000 - step * (1000 // num_steps)  # 时间步从1000递减
    hidden, enc_hidden, timestep = make_inputs(t, noise_level=0.02, base_hidden=base_hidden)

    start = time.time()
    with torch.no_grad():
        out = transformer(hidden_states=hidden, encoder_hidden_states=enc_hidden, timestep=timestep)
        if hasattr(out, 'sample'):
            result = out.sample
        else:
            result = out[0] if isinstance(out, tuple) else out
    elapsed = time.time() - start

    baseline_outputs.append(result.detach().cpu())
    baseline_times.append(elapsed)
    base_hidden = hidden  # 下一步基于这步的输入

    if step == 0:
        print(f"  输出形状: {result.shape}")

print(f"  总耗时: {sum(baseline_times):.2f}s")
print(f"  平均每步: {np.mean(baseline_times):.3f}s")
print()


# ============================================================
# 实验 2: FirstBlockCache 标准缓存（fp32）
# ============================================================

print("=" * 60)
print("实验 2: FirstBlockCache（fp32 缓存）")
print("=" * 60)

# 手动实现 FirstBlockCache 逻辑
# 因为我们直接调用 transformer，不走 pipeline，需要手动实现缓存逻辑

class ManualFirstBlockCache:
    """手动实现的 FirstBlockCache（不依赖 diffusers hook 系统）"""

    def __init__(self, model, threshold=0.05):
        self.model = model
        self.threshold = threshold
        self.head_block_residual = None
        self.cached_output = None
        self.cache_hits = 0
        self.total_steps = 0
        self.cache_memory_bytes = 0

    def run_step(self, hidden_states, encoder_hidden_states, timestep):
        self.total_steps += 1

        # 通过第一个 block 获取残差（简化：直接用完整模型输出做近似）
        with torch.no_grad():
            output = self.model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
            )
            if hasattr(output, 'sample'):
                result = output.sample
            else:
                result = output[0] if isinstance(output, tuple) else output

        # 计算残差
        current_residual = result - hidden_states[:, :result.shape[1]] if result.shape[1] <= hidden_states.shape[1] else result

        # 判断是否可以复用缓存
        if self.head_block_residual is not None:
            # 计算差异（和 FirstBlockCache 相同的指标）
            diff_tensor = current_residual.detach().cpu().float()
            prev_tensor = self.head_block_residual.float()

            absmean = (diff_tensor - prev_tensor).abs().mean()
            prev_absmean = prev_tensor.abs().mean()

            if prev_absmean > 1e-10:
                diff = (absmean / prev_absmean).item()
            else:
                diff = float('inf')

            if diff < self.threshold:
                # 缓存命中：复用上次结果
                self.cache_hits += 1
                return self.cached_output.to(result.device)

        # 缓存未命中：存储结果
        self.head_block_residual = current_residual.detach().cpu()
        self.cached_output = result.detach().cpu()
        self.cache_memory_bytes = self.head_block_residual.nelement() * 4 + self.cached_output.nelement() * 4
        return result


fbc = ManualFirstBlockCache(transformer, threshold=0.05)
base_hidden = None
fbc_outputs = []
fbc_times = []

for step in range(num_steps):
    t = 1000 - step * (1000 // num_steps)
    hidden, enc_hidden, timestep = make_inputs(t, noise_level=0.02, base_hidden=base_hidden)

    start = time.time()
    result = fbc.run_step(hidden, enc_hidden, timestep)
    elapsed = time.time() - start

    fbc_outputs.append(result.detach().cpu())
    fbc_times.append(elapsed)
    base_hidden = hidden

print(f"  缓存命中: {fbc.cache_hits}/{fbc.total_steps} ({fbc.cache_hits/fbc.total_steps*100:.0f}%)")
print(f"  总耗时: {sum(fbc_times):.2f}s")
print(f"  平均每步: {np.mean(fbc_times):.3f}s")
print(f"  缓存显存 (fp32): {fbc.cache_memory_bytes / 1024:.1f} KB")
print()


# ============================================================
# 实验 3: FirstBlockCache + TurboQuant 压缩
# ============================================================

print("=" * 60)
print("实验 3: FirstBlockCache + TurboQuant 压缩缓存")
print("=" * 60)

class CompressedFirstBlockCache:
    """FirstBlockCache + TurboQuant 压缩版本"""

    def __init__(self, model, threshold=0.05, bits=3):
        self.model = model
        self.threshold = threshold
        self.bits = bits
        self.head_block_residual = None  # 保持 fp32 用于阈值判断
        self.compressed_output = None
        self.compressor = None
        self.cache_hits = 0
        self.total_steps = 0
        self.compressed_bytes = 0
        self.original_bytes = 0
        self.compress_time = 0
        self.decompress_time = 0

    def run_step(self, hidden_states, encoder_hidden_states, timestep):
        self.total_steps += 1

        with torch.no_grad():
            output = self.model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
            )
            if hasattr(output, 'sample'):
                result = output.sample
            else:
                result = output[0] if isinstance(output, tuple) else output

        current_residual = result - hidden_states[:, :result.shape[1]] if result.shape[1] <= hidden_states.shape[1] else result

        # 判断是否可以复用
        if self.head_block_residual is not None:
            diff_tensor = current_residual.detach().cpu().float()
            prev_tensor = self.head_block_residual.float()
            absmean = (diff_tensor - prev_tensor).abs().mean()
            prev_absmean = prev_tensor.abs().mean()

            if prev_absmean > 1e-10:
                diff = (absmean / prev_absmean).item()
            else:
                diff = float('inf')

            if diff < self.threshold:
                # 缓存命中：解压缓存
                self.cache_hits += 1
                t0 = time.time()
                decompressed = self._decompress_output(result.shape, result.device, result.dtype)
                self.decompress_time += time.time() - t0
                return decompressed

        # 缓存未命中：压缩存储
        self.head_block_residual = current_residual.detach().cpu()

        t0 = time.time()
        self._compress_output(result)
        self.compress_time += time.time() - t0

        return result

    def _compress_output(self, tensor):
        """用 PolarQuant 压缩输出"""
        # 展平为 2D: (batch*spatial, dim)
        self.original_shape = tensor.shape
        flat = tensor.detach().cpu().float().reshape(-1, tensor.shape[-1])

        d = flat.shape[-1]
        if self.compressor is None or self.compressor.dim != d:
            self.compressor = PolarQuantCompressor(dim=d, bits=self.bits)

        self.compressed_output = self.compressor.compress_tensor(flat)
        self.compressed_bytes = self.compressor.compressed_size_bytes(self.compressed_output)
        self.original_bytes = flat.nelement() * 4  # fp32

    def _decompress_output(self, target_shape, device, dtype):
        """解压输出"""
        flat = self.compressor.decompress_tensor(self.compressed_output, device="cpu", dtype=torch.float32)
        return flat.reshape(self.original_shape).to(device=device, dtype=dtype)


# 测试不同 bit 配置
for bits in [2, 3, 4]:
    print(f"\n--- {bits}-bit 压缩 ---")

    cfbc = CompressedFirstBlockCache(transformer, threshold=0.05, bits=bits)
    base_hidden = None
    cfbc_outputs = []
    cfbc_times = []

    for step in range(num_steps):
        t = 1000 - step * (1000 // num_steps)
        hidden, enc_hidden, timestep = make_inputs(t, noise_level=0.02, base_hidden=base_hidden)

        start = time.time()
        result = cfbc.run_step(hidden, enc_hidden, timestep)
        elapsed = time.time() - start

        cfbc_outputs.append(result.detach().cpu())
        cfbc_times.append(elapsed)
        base_hidden = hidden

    print(f"  缓存命中: {cfbc.cache_hits}/{cfbc.total_steps} ({cfbc.cache_hits/cfbc.total_steps*100:.0f}%)")
    print(f"  总耗时: {sum(cfbc_times):.2f}s")
    print(f"  平均每步: {np.mean(cfbc_times):.3f}s")

    if cfbc.original_bytes > 0:
        ratio = cfbc.original_bytes / cfbc.compressed_bytes
        print(f"  缓存显存 (fp32): {cfbc.original_bytes / 1024:.1f} KB")
        print(f"  缓存显存 ({bits}bit): {cfbc.compressed_bytes / 1024:.1f} KB")
        print(f"  压缩比: {ratio:.1f}×")
        print(f"  压缩耗时: {cfbc.compress_time:.3f}s")
        print(f"  解压耗时: {cfbc.decompress_time:.3f}s")

    # 计算和 baseline 的质量差异
    if len(cfbc_outputs) == len(baseline_outputs):
        cos_sims = []
        mses = []
        for bo, co in zip(baseline_outputs, cfbc_outputs):
            bo_flat = bo.float().flatten().numpy()
            co_flat = co.float().flatten().numpy()
            cos = np.dot(bo_flat, co_flat) / (np.linalg.norm(bo_flat) * np.linalg.norm(co_flat) + 1e-10)
            mse = np.mean((bo_flat - co_flat) ** 2)
            cos_sims.append(cos)
            mses.append(mse)
        print(f"  vs Baseline 余弦相似度: {np.mean(cos_sims):.4f}")
        print(f"  vs Baseline MSE: {np.mean(mses):.6f}")

print()

# ============================================================
# 汇总对比
# ============================================================

print("=" * 60)
print("汇总对比")
print("=" * 60)

# 重新跑一次收集所有数据
configs_results = {}

# Baseline
configs_results["baseline"] = {
    "time": sum(baseline_times),
    "avg_step": np.mean(baseline_times),
    "cache_kb": 0,
    "cos_sim": 1.0,
}

# FBC only
fbc_cos = []
for bo, fo in zip(baseline_outputs, fbc_outputs):
    bo_flat = bo.float().flatten().numpy()
    fo_flat = fo.float().flatten().numpy()
    cos = np.dot(bo_flat, fo_flat) / (np.linalg.norm(bo_flat) * np.linalg.norm(fo_flat) + 1e-10)
    fbc_cos.append(cos)

configs_results["fbc_fp32"] = {
    "time": sum(fbc_times),
    "avg_step": np.mean(fbc_times),
    "cache_kb": fbc.cache_memory_bytes / 1024,
    "cos_sim": np.mean(fbc_cos),
}

print(f"{'配置':<20s} {'总耗时':>8s} {'每步':>8s} {'缓存':>10s} {'vs Baseline':>12s}")
print("-" * 62)
for name, r in configs_results.items():
    print(f"{name:<20s} {r['time']:>7.2f}s {r['avg_step']:>7.3f}s {r['cache_kb']:>8.1f}KB {r['cos_sim']:>11.4f}")

print()
print("=" * 60)
print("结论")
print("=" * 60)
print("如果 TurboQuant 压缩后：")
print("  1. 缓存大小显著减小（压缩比 > 3×）")
print("  2. vs Baseline 余弦相似度 > 0.95")
print("  3. 解压耗时远小于跳过计算节省的时间")
print("→ 则融合方案有效，值得进一步优化")
