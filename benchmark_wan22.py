"""
在 Wan2.2-T2V-A14B 上验证 VedioQuant（只下载 transformer 部分）
"""

import torch
import numpy as np
import time
import sys
sys.path.insert(0, '.')
from vedioquant.compressor.turbo_quant import TurboQuantCompressor

def log(msg):
    print(msg, flush=True)

log("=" * 70)
log("  VedioQuant on Wan2.2 (Transformer Only)")
log("=" * 70)

# 先试 5B，如果下载失败试 14B 的 transformer subfolder
model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
log(f"\n加载 {model_id} transformer...")

from diffusers.models import WanTransformer3DModel

try:
    transformer = WanTransformer3DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=torch.float32,
    )
    log("5B transformer 加载成功")
except Exception as e:
    log(f"5B 加载失败: {e}")
    log("尝试 14B transformer...")
    model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    try:
        transformer = WanTransformer3DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=torch.float32,
        )
        log("14B transformer 加载成功")
    except Exception as e2:
        log(f"14B 也失败: {e2}")
        log("退出")
        sys.exit(1)

transformer = transformer.to("mps")
transformer.eval()

config = transformer.config
hidden_dim = config.num_attention_heads * config.attention_head_dim
num_layers = config.num_layers
in_channels = config.in_channels
log(f"模型: {num_layers} 层, hidden_dim={hidden_dim}, in_channels={in_channels}")
log(f"来源: {model_id}")

# 测试
tests = [
    {"name": "小规模", "lat_t": 1, "lat_h": 16, "lat_w": 16},
    {"name": "中等规模", "lat_t": 3, "lat_h": 30, "lat_w": 52},
]

for test in tests:
    lat_t, lat_h, lat_w = test["lat_t"], test["lat_h"], test["lat_w"]
    tokens = lat_t * (lat_h // 2) * (lat_w // 2)

    log(f"\n{'='*70}")
    log(f"  {test['name']}: latent=({lat_t},{lat_h},{lat_w}), tokens={tokens}")
    log(f"{'='*70}")

    torch.manual_seed(42)
    hidden = torch.randn(1, in_channels, lat_t, lat_h, lat_w, dtype=torch.float32, device="mps")
    enc = torch.randn(1, 77, 4096, dtype=torch.float32, device="mps")
    ts = torch.tensor([500.0], dtype=torch.float32, device="mps")

    captured = []
    def hook_fn(module, input, output):
        if len(captured) < 1:
            out = output[0] if isinstance(output, tuple) else output
            if isinstance(out, torch.Tensor):
                captured.append(out.detach().cpu().float())

    hook = None
    for name, module in transformer.named_modules():
        if "attn1" in name and "to_" not in name and "norm" not in name:
            hook = module.register_forward_hook(hook_fn)
            break

    log(f"  前向传播中...")
    start = time.time()
    try:
        with torch.no_grad():
            transformer(hidden_states=hidden, encoder_hidden_states=enc, timestep=ts)
        log(f"  前向传播: {time.time()-start:.2f}s")
    except Exception as e:
        log(f"  失败: {e}")
        if hook: hook.remove()
        continue

    if hook: hook.remove()

    if not captured:
        log("  未捕获特征")
        continue

    attn = captured[0]
    log(f"  捕获注意力输出: {attn.shape}")

    flat = attn.reshape(-1, attn.shape[-1])
    N, d = flat.shape
    log(f"  展平: ({N}, {d})")

    orig_bytes = N * d * 4
    orig_mb = orig_bytes / 1024**2
    log(f"\n  --- 原始 fp32 ---")
    log(f"  单层: {orig_mb:.2f} MB | {num_layers}层: {orig_mb*num_layers/1024:.2f} GB")

    # 旋转高斯化验证
    from scipy.stats import kurtosis
    comp_test = TurboQuantCompressor(dim=d, bits=3)
    sample = flat[0].numpy()
    gamma = np.linalg.norm(sample)
    if gamma > 1e-10:
        x_hat = sample / gamma
        Pi = comp_test.polar.Pi.numpy()
        x_rot = Pi @ x_hat
        log(f"\n  --- 旋转高斯化验证 ---")
        log(f"  旋转前峰度: {kurtosis(x_hat):.1f}")
        log(f"  旋转后峰度: {kurtosis(x_rot):.1f} (目标≈0)")
        log(f"  旋转后标准差: {x_rot.std():.6f} (理论值: {1/np.sqrt(d):.6f})")

    for bits in [2, 3, 4]:
        comp = TurboQuantCompressor(dim=d, bits=bits)

        t0 = time.time()
        compressed = comp.compress(flat)
        ct = time.time() - t0

        t0 = time.time()
        recon = comp.decompress(compressed)
        dt = time.time() - t0

        comp_bytes = comp.compressed_size_bytes(flat)
        comp_mb = comp_bytes / 1024**2
        ratio = orig_bytes / comp_bytes

        n_test = min(100, N)
        cos_list = []
        for i in range(n_test):
            v = flat[i].numpy()
            r = recon[i].numpy()
            cos = np.dot(v, r) / (np.linalg.norm(v) * np.linalg.norm(r) + 1e-10)
            cos_list.append(cos)

        log(f"\n  --- {bits}-bit TurboQuant ---")
        log(f"  单层: {orig_mb:.2f} MB → {comp_mb:.2f} MB (压缩比 {ratio:.1f}×)")
        log(f"  {num_layers}层: {orig_mb*num_layers/1024:.2f} GB → {comp_mb*num_layers/1024:.2f} GB")
        log(f"  压缩: {ct:.3f}s | 解压: {dt:.3f}s")
        log(f"  余弦: 平均 {np.mean(cos_list):.4f}, 最低 {np.min(cos_list):.4f}")

    # 外推 720P/81帧
    tokens_720 = 21 * 45 * 80
    log(f"\n  --- 外推 720P/81帧 (3-bit, {num_layers}层全缓存) ---")
    fp32_720 = tokens_720 * d * 4 * num_layers
    bit3_720 = tokens_720 * (2 + d * 3 / 8) * num_layers + 8 * 4 * num_layers
    log(f"  fp32:  {fp32_720/1024**3:.2f} GB")
    log(f"  3-bit: {bit3_720/1024**3:.2f} GB")
    log(f"  节省:  {(fp32_720-bit3_720)/1024**3:.2f} GB")

    del hidden, enc, captured, attn, flat, recon

log(f"\n{'='*70}")
log("  Wan2.2 验证完成!")
log(f"{'='*70}")
