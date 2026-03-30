"""
第二阶段（轻量版）：只下载 Wan2.1 Transformer，提取注意力特征
跳过完整 pipeline（T5 文本编码器太大），直接加载 transformer 用随机输入提取特征。

目的：拿到真实的注意力层中间张量，分析分布 + 测试 TurboQuant 压缩质量。
不需要生成真实视频，只需要特征的统计特性。
"""

import torch
import numpy as np
from scipy.stats import kurtosis
import time

print("=" * 60)
print("第二阶段（轻量版）：直接加载 Transformer 提取特征")
print("=" * 60)
print()

# ============================================================
# 第一步：只下载 Transformer 部分（~2.6GB）
# ============================================================
print("第一步：加载 Wan2.1 Transformer（~2.6GB）")
print()

from diffusers.models import WanTransformer3DModel

start = time.time()
transformer = WanTransformer3DModel.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    subfolder="transformer",
    torch_dtype=torch.float32,
)
transformer = transformer.to("mps")
transformer.eval()
print(f"加载完成，耗时 {time.time()-start:.0f} 秒")
print()

# 打印模型结构
print("模型结构（注意力相关模块）:")
attn_modules = []
for name, module in transformer.named_modules():
    if "attn" in name.lower() and name.count(".") <= 3:
        print(f"  {name}: {type(module).__name__}")
        attn_modules.append((name, module))
print(f"\n共找到 {len(attn_modules)} 个注意力相关模块")
print()

# ============================================================
# 第二步：Hook 注意力层
# ============================================================
print("=" * 60)
print("第二步：Hook 注意力层，捕获中间特征")
print("=" * 60)

captured_features = []
hooks = []

def make_hook(name):
    def hook_fn(module, input, output):
        if len(captured_features) < 20:
            if isinstance(output, torch.Tensor):
                captured_features.append({
                    "name": name,
                    "shape": tuple(output.shape),
                    "data": output.detach().cpu().float()
                })
            elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                captured_features.append({
                    "name": name,
                    "shape": tuple(output[0].shape),
                    "data": output[0].detach().cpu().float()
                })
    return hook_fn

# Hook 所有注意力层
for name, module in transformer.named_modules():
    if "attn1" in name and "to_" not in name and "norm" not in name:
        hook = module.register_forward_hook(make_hook(name))
        hooks.append(hook)
    elif "attn2" in name and "to_" not in name and "norm" not in name:
        hook = module.register_forward_hook(make_hook(name))
        hooks.append(hook)

if len(hooks) == 0:
    # fallback: hook any attention-like module
    for name, module in transformer.named_modules():
        if "attn" in name.lower() and "norm" not in name and "proj" not in name and "to_" not in name:
            if name.count(".") <= 2:
                hook = module.register_forward_hook(make_hook(name))
                hooks.append(hook)
                if len(hooks) >= 10:
                    break

print(f"注册了 {len(hooks)} 个 hooks")
print()

# ============================================================
# 第三步：构造模拟输入，前向传播
# ============================================================
print("=" * 60)
print("第三步：构造输入，运行前向传播")
print("=" * 60)

# 查看 transformer 的 forward 签名
import inspect
sig = inspect.signature(transformer.forward)
print(f"forward 参数: {list(sig.parameters.keys())}")
print()

# 构造模拟输入
# Wan2.1 transformer 通常需要: hidden_states, encoder_hidden_states, timestep
# hidden_states: (batch, channels, frames, height, width) 或 (batch, seq_len, dim)
# 我们需要查看模型 config 来确定维度

config = transformer.config
print(f"模型配置:")
for key in ['in_channels', 'num_attention_heads', 'attention_head_dim', 'num_layers',
            'cross_attention_dim', 'sample_size', 'patch_size']:
    if hasattr(config, key):
        print(f"  {key}: {getattr(config, key)}")

print()

try:
    in_channels = getattr(config, 'in_channels', 16)
    # text_embedder 的输入是 4096 维 (T5 encoder output dim)
    cross_dim = 4096
    num_heads = getattr(config, 'num_attention_heads', 12)
    head_dim = getattr(config, 'attention_head_dim', 128)

    # 小尺寸输入，只为拿特征
    batch = 1
    frames = 1
    height = 16  # latent space, patch_size=[1,2,2] 所以要能整除2
    width = 16

    hidden_states = torch.randn(batch, in_channels, frames, height, width,
                                 dtype=torch.float32, device="mps")

    # encoder_hidden_states (来自 T5 文本编码器，dim=4096)
    seq_len = 32
    encoder_hidden_states = torch.randn(batch, seq_len, cross_dim,
                                         dtype=torch.float32, device="mps")

    # timestep
    timestep = torch.tensor([500.0], dtype=torch.float32, device="mps")

    print(f"输入形状:")
    print(f"  hidden_states: {hidden_states.shape}")
    print(f"  encoder_hidden_states: {encoder_hidden_states.shape}")
    print(f"  timestep: {timestep.shape}")
    print()

    print("运行前向传播...")
    start = time.time()
    with torch.no_grad():
        try:
            output = transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
            )
            print(f"前向传播完成，耗时 {time.time()-start:.1f} 秒")
        except Exception as e:
            print(f"标准输入失败: {e}")
            print("尝试调整输入格式...")
            # 有些模型需要不同的输入格式
            output = transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                return_dict=False,
            )
            print(f"前向传播完成，耗时 {time.time()-start:.1f} 秒")

except Exception as e:
    print(f"前向传播出错: {e}")
    print()
    print("尝试分析模型结构来确定正确的输入格式...")
    # 打印更详细的结构
    for name, param in list(transformer.named_parameters())[:10]:
        print(f"  {name}: {param.shape}")

# 移除 hooks
for h in hooks:
    h.remove()

print(f"\n捕获到 {len(captured_features)} 个特征张量")
print()

# ============================================================
# 第四步：分析特征分布
# ============================================================
if len(captured_features) > 0:
    print("=" * 60)
    print("第四步：特征分布分析")
    print("=" * 60)

    for i, feat_info in enumerate(captured_features[:8]):
        feat = feat_info["data"]
        feat_np = feat.numpy().flatten()
        sample = feat_np[:min(100000, len(feat_np))]

        k = kurtosis(sample)
        print(f"\n特征 #{i}: {feat_info['name']}")
        print(f"  形状: {feat_info['shape']}")
        print(f"  均值: {sample.mean():.6f}")
        print(f"  标准差: {sample.std():.6f}")
        print(f"  最小值: {sample.min():.4f}, 最大值: {sample.max():.4f}")
        print(f"  峰度: {k:.1f}  (高斯=0, 语言模型KV≈900)")

    # 保存特征
    torch.save(captured_features, "video_features.pt")
    print(f"\n特征已保存到 video_features.pt")

    # ============================================================
    # 第五步：TurboQuant 压缩测试
    # ============================================================
    print()
    print("=" * 60)
    print("第五步：TurboQuant 压缩质量测试")
    print("=" * 60)

    # 取第一个特征
    feat = captured_features[0]["data"]
    feat_flat = feat.reshape(-1, feat.shape[-1]).numpy().astype(np.float64)
    d = feat_flat.shape[-1]
    num_tokens = feat_flat.shape[0]

    print(f"特征维度 d = {d}, Token 数 = {num_tokens}")

    # PolarQuant 工具函数
    def lloyd_max_quantizer(data, num_levels, max_iter=100):
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

    def quantize(data, levels, boundaries):
        all_boundaries = np.concatenate([[-np.inf], boundaries, [np.inf]])
        result = np.zeros_like(data)
        for i in range(len(levels)):
            mask = (data >= all_boundaries[i]) & (data < all_boundaries[i + 1])
            result[mask] = levels[i]
        return result

    def test_polarquant(vec, bits, Pi):
        gamma = np.linalg.norm(vec)
        if gamma < 1e-10:
            return 1.0, 0.0
        x_hat = vec / gamma
        x_rotated = Pi @ x_hat
        levels, boundaries = lloyd_max_quantizer(x_rotated, 2 ** bits)
        x_quantized = quantize(x_rotated, levels, boundaries)
        x_recon = Pi.T @ x_quantized * gamma
        cos_sim = np.dot(vec, x_recon) / (np.linalg.norm(vec) * np.linalg.norm(x_recon) + 1e-10)
        mse = np.mean((vec - x_recon) ** 2)
        return cos_sim, mse

    # 生成旋转矩阵
    print(f"\n生成 {d}×{d} 随机正交矩阵...")
    np.random.seed(42)
    Pi, _ = np.linalg.qr(np.random.randn(d, d))

    # 旋转高斯化验证
    sample_vec = feat_flat[0]
    gamma = np.linalg.norm(sample_vec)
    if gamma > 1e-10:
        x_hat = sample_vec / gamma
        x_rotated = Pi @ x_hat
        print(f"\n旋转高斯化验证:")
        print(f"  旋转前峰度: {kurtosis(x_hat):.1f}")
        print(f"  旋转后峰度: {kurtosis(x_rotated):.1f}  (目标≈0)")
        print(f"  旋转后标准差: {x_rotated.std():.6f}")
        print(f"  理论值 1/√d:  {1/np.sqrt(d):.6f}")

    # 测试不同 bit 配置
    num_test = min(50, num_tokens)
    configs = [2, 3, 4]

    print(f"\n对 {num_test} 个 token 测试压缩质量:")
    print(f"{'配置':<15s} {'平均余弦相似度':>12s} {'平均MSE':>12s}")
    print("-" * 42)

    for bits in configs:
        cos_list, mse_list = [], []
        for j in range(num_test):
            cs, ms = test_polarquant(feat_flat[j], bits, Pi)
            cos_list.append(cs)
            mse_list.append(ms)
        print(f"{bits}-bit{'':<10s} {np.mean(cos_list):>12.4f} {np.mean(mse_list):>12.6f}")

    print()
    print("=" * 60)
    print("判断标准:")
    print("  余弦相似度 > 0.90 → 可以进入第三阶段")
    print("  余弦相似度 0.80-0.90 → 需要调整参数")
    print("  余弦相似度 < 0.80 → 分布差异太大")
    print("=" * 60)

else:
    print("未捕获到特征。打印完整模型结构用于调试:")
    for name, module in transformer.named_modules():
        if name:
            print(f"  {name}: {type(module).__name__}")
