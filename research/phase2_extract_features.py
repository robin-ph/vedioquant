"""
第二阶段：从 Wan2.1-1.3B 提取注意力层特征
- 下载模型（首次约2.6GB）
- Hook 注意力层，捕获 K、V 向量
- 保存特征 + 分析分布
- 测试 TurboQuant 压缩质量
"""

import torch
import numpy as np
from scipy.stats import kurtosis
import time

# ============================================================
# 第一步：加载模型
# ============================================================
print("=" * 60)
print("第一步：加载 Wan2.1-T2V-1.3B 模型")
print("=" * 60)
print("首次运行会下载模型（约2.6GB），请耐心等待...")
print()

from diffusers import WanPipeline

start = time.time()
pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    torch_dtype=torch.float32,  # MPS 对 float16 支持有限，用 float32 更稳
)
pipe = pipe.to("mps")
print(f"模型加载完成，耗时 {time.time()-start:.0f} 秒")
print()

# ============================================================
# 第二步：Hook 注意力层，捕获特征
# ============================================================
print("=" * 60)
print("第二步：Hook 注意力层")
print("=" * 60)

# 查看模型结构，找到注意力层
transformer = pipe.transformer
print(f"Transformer 类型: {type(transformer).__name__}")

# 打印模型结构的前几层，找到注意力模块
print("\n模型结构（前3层）:")
for name, module in list(transformer.named_modules())[:30]:
    if name:
        print(f"  {name}: {type(module).__name__}")

# 捕获字典：记录每个 hook 点的特征
captured = {
    "keys": [],      # K 向量
    "values": [],    # V 向量
    "outputs": [],   # 注意力输出
}

hooks = []

def make_hook(capture_list, name):
    """创建 hook 函数，捕获模块的输入输出"""
    def hook_fn(module, input, output):
        # 只捕获前几次，避免显存爆炸
        if len(capture_list) < 10:
            if isinstance(output, torch.Tensor):
                capture_list.append({
                    "name": name,
                    "shape": output.shape,
                    "data": output.detach().cpu().float()
                })
            elif isinstance(output, tuple) and len(output) > 0:
                capture_list.append({
                    "name": name,
                    "shape": output[0].shape,
                    "data": output[0].detach().cpu().float()
                })
    return hook_fn

# 尝试在多个可能的注意力层位置注册 hook
hook_registered = False
for name, module in transformer.named_modules():
    # 常见的注意力层命名模式
    if any(key in name.lower() for key in ["attn", "attention", "self_attn"]):
        if not any(sub in name for sub in ["norm", "dropout", "proj"]):
            hook = module.register_forward_hook(make_hook(captured["outputs"], name))
            hooks.append(hook)
            hook_registered = True
            if len(hooks) >= 5:  # 最多 hook 5 个注意力层
                break

if not hook_registered:
    # 如果没找到注意力层，hook 整个 transformer block
    print("未找到标准注意力层，尝试 hook transformer blocks...")
    for name, module in transformer.named_modules():
        if "block" in name.lower() and name.count(".") <= 1:
            hook = module.register_forward_hook(make_hook(captured["outputs"], name))
            hooks.append(hook)
            if len(hooks) >= 3:
                break

print(f"\n注册了 {len(hooks)} 个 hooks")
for h_name in [n for n, _ in transformer.named_modules() if any(k in n.lower() for k in ["attn", "attention", "block"])][:5]:
    print(f"  → {h_name}")
print()

# ============================================================
# 第三步：运行推理，捕获特征
# ============================================================
print("=" * 60)
print("第三步：运行推理（5步，仅为提取特征）")
print("=" * 60)

start = time.time()
try:
    with torch.no_grad():
        output = pipe(
            prompt="a cat sitting on a sofa",
            num_inference_steps=5,
            height=480,
            width=832,
            num_frames=17,
            guidance_scale=5.0,
        )
    print(f"推理完成，耗时 {time.time()-start:.0f} 秒")
except Exception as e:
    print(f"推理出错: {e}")
    print("尝试降低分辨率...")
    try:
        with torch.no_grad():
            output = pipe(
                prompt="a cat sitting on a sofa",
                num_inference_steps=5,
                height=256,
                width=256,
                num_frames=9,
                guidance_scale=5.0,
            )
        print(f"低分辨率推理完成，耗时 {time.time()-start:.0f} 秒")
    except Exception as e2:
        print(f"低分辨率也失败: {e2}")

# 移除 hooks
for h in hooks:
    h.remove()

print(f"\n捕获到 {len(captured['outputs'])} 个特征张量")
print()

# ============================================================
# 第四步：分析特征分布
# ============================================================
if len(captured["outputs"]) > 0:
    print("=" * 60)
    print("第四步：特征分布分析")
    print("=" * 60)

    for i, feat_info in enumerate(captured["outputs"][:5]):
        feat = feat_info["data"]
        feat_np = feat.numpy().flatten()

        # 只取前100k个值做统计（避免太慢）
        sample = feat_np[:min(100000, len(feat_np))]

        print(f"\n特征 #{i}: {feat_info['name']}")
        print(f"  形状: {feat_info['shape']}")
        print(f"  均值: {sample.mean():.6f}")
        print(f"  标准差: {sample.std():.6f}")
        print(f"  最小值: {sample.min():.6f}")
        print(f"  最大值: {sample.max():.6f}")
        print(f"  峰度: {kurtosis(sample):.1f}  (高斯=0, 语言模型KV≈900)")

    # 保存特征供后续测试
    save_path = "video_features.pt"
    torch.save(captured["outputs"], save_path)
    print(f"\n特征已保存到 {save_path}")

    # ============================================================
    # 第五步：对视频特征跑 TurboQuant 压缩测试
    # ============================================================
    print()
    print("=" * 60)
    print("第五步：TurboQuant 压缩质量测试")
    print("=" * 60)

    feat = captured["outputs"][0]["data"]
    feat_flat = feat.reshape(-1, feat.shape[-1])  # (tokens, dim)
    d = feat_flat.shape[-1]

    print(f"特征维度 d = {d}")
    print(f"Token 数 = {feat_flat.shape[0]}")
    print()

    # --- 复用 Step 1 的 PolarQuant ---
    from scipy.stats import kurtosis as kurt_func

    def lloyd_max_quantizer(data, num_levels, max_iter=100):
        min_val, max_val = data.min(), data.max()
        levels = np.linspace(min_val, max_val, num_levels)
        for _ in range(max_iter):
            boundaries = (levels[:-1] + levels[1:]) / 2
            all_boundaries = np.concatenate([[-np.inf], boundaries, [np.inf]])
            new_levels = np.zeros(num_levels)
            for i in range(num_levels):
                mask = (data >= all_boundaries[i]) & (data < all_boundaries[i + 1])
                if mask.sum() > 0:
                    new_levels[i] = data[mask].mean()
                else:
                    new_levels[i] = levels[i]
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

    def test_compression(vec_np, bits_step1, Pi):
        """对单个向量测试 PolarQuant 压缩质量"""
        # 范数分离
        gamma = np.linalg.norm(vec_np)
        if gamma < 1e-10:
            return 1.0, 0.0  # 零向量
        x_hat = vec_np / gamma

        # 旋转
        x_rotated = Pi @ x_hat

        # 量化
        num_levels = 2 ** bits_step1
        levels, boundaries = lloyd_max_quantizer(x_rotated, num_levels)
        x_quantized = quantize(x_rotated, levels, boundaries)

        # 反旋转重建
        x_recon = Pi.T @ x_quantized * gamma

        # 余弦相似度
        cos_sim = np.dot(vec_np, x_recon) / (np.linalg.norm(vec_np) * np.linalg.norm(x_recon) + 1e-10)
        mse = np.mean((vec_np - x_recon) ** 2)
        return cos_sim, mse

    # 生成旋转矩阵（用实际维度）
    print(f"生成 {d}×{d} 随机正交矩阵...")
    np.random.seed(42)
    random_matrix = np.random.randn(d, d)
    Pi, _ = np.linalg.qr(random_matrix)

    # 先检查旋转高斯化效果
    sample_vec = feat_flat[0].numpy()
    gamma = np.linalg.norm(sample_vec)
    if gamma > 1e-10:
        x_hat = sample_vec / gamma
        x_rotated = Pi @ x_hat
        print(f"\n旋转高斯化验证（第一个token）:")
        print(f"  旋转前峰度: {kurt_func(x_hat):.1f}")
        print(f"  旋转后峰度: {kurt_func(x_rotated):.1f}  (目标≈0)")
        print(f"  旋转后标准差: {x_rotated.std():.6f}")
        print(f"  理论值 1/√d:  {1/np.sqrt(d):.6f}")
    print()

    # 对多个 token 测试不同 bit 配置
    num_test_tokens = min(50, feat_flat.shape[0])
    test_configs = [
        {"bits": 2, "label": "2-bit (Step1=2bit)"},
        {"bits": 3, "label": "3-bit (Step1=3bit)"},
        {"bits": 4, "label": "4-bit (Step1=4bit)"},
    ]

    print(f"对 {num_test_tokens} 个 token 测试压缩质量:")
    print(f"{'配置':<25s} {'平均余弦相似度':>12s} {'平均MSE':>12s}")
    print("-" * 52)

    for config in test_configs:
        cos_sims = []
        mses = []
        for j in range(num_test_tokens):
            vec = feat_flat[j].numpy().astype(np.float64)
            cs, mse = test_compression(vec, config["bits"], Pi)
            cos_sims.append(cs)
            mses.append(mse)

        avg_cos = np.mean(cos_sims)
        avg_mse = np.mean(mses)
        print(f"{config['label']:<25s} {avg_cos:>12.4f} {avg_mse:>12.6f}")

    print()
    print("=" * 60)
    print("判断标准:")
    print("  余弦相似度 > 0.90 → 可以进入第三阶段")
    print("  余弦相似度 0.80-0.90 → 需要调整参数")
    print("  余弦相似度 < 0.80 → 分布差异太大")
    print("=" * 60)

else:
    print("未捕获到特征，请检查模型结构和 hook 位置")
    print("尝试打印完整模型结构:")
    for name, module in transformer.named_modules():
        if name:
            print(f"  {name}")
