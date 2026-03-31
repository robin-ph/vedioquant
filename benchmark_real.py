"""
实测：真实规模缓存压缩效果
用多个递增规模测试，flush 输出防止缓冲
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
log("  VedioQuant 实测 Benchmark")
log("=" * 70)

# 加载模型
log("\n加载 Wan2.1-1.3B Transformer...")
from diffusers.models import WanTransformer3DModel

transformer = WanTransformer3DModel.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    subfolder="transformer",
    torch_dtype=torch.float32,
)
log("加载到 MPS...")
transformer = transformer.to("mps")
transformer.eval()
log("模型就绪")

hidden_dim = 1536
in_channels = 16

# ============================================================
# 多个规模测试
# ============================================================

tests = [
    {"name": "小规模 (验证流程)", "lat_t": 1, "lat_h": 16, "lat_w": 16},
    {"name": "中等规模",         "lat_t": 3, "lat_h": 30, "lat_w": 52},
    {"name": "480P/17帧等效",    "lat_t": 5, "lat_h": 60, "lat_w": 104},
]

for test in tests:
    lat_t, lat_h, lat_w = test["lat_t"], test["lat_h"], test["lat_w"]
    tokens = (lat_t) * (lat_h // 2) * (lat_w // 2)

    log(f"\n{'='*70}")
    log(f"  {test['name']}: latent=({lat_t},{lat_h},{lat_w}), tokens={tokens}")
    log(f"{'='*70}")

    torch.manual_seed(42)
    hidden = torch.randn(1, in_channels, lat_t, lat_h, lat_w, dtype=torch.float32, device="mps")
    enc = torch.randn(1, 77, 4096, dtype=torch.float32, device="mps")
    ts = torch.tensor([500.0], dtype=torch.float32, device="mps")

    # Hook 注意力层
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

    # 前向传播
    log(f"  前向传播中...")
    start = time.time()
    try:
        with torch.no_grad():
            transformer(hidden_states=hidden, encoder_hidden_states=enc, timestep=ts)
        fwd_time = time.time() - start
        log(f"  前向传播: {fwd_time:.2f}s")
    except Exception as e:
        log(f"  失败: {e}")
        if hook: hook.remove()
        continue

    if hook: hook.remove()

    if not captured:
        log("  未捕获特征，跳过")
        continue

    attn = captured[0]
    log(f"  捕获注意力输出: {attn.shape}")

    flat = attn.reshape(-1, attn.shape[-1])
    N, d = flat.shape
    log(f"  展平: ({N}, {d})")

    # 原始大小
    orig_bytes = N * d * 4
    orig_mb = orig_bytes / 1024**2
    log(f"\n  --- 原始 fp32 ---")
    log(f"  单层: {orig_mb:.2f} MB")
    log(f"  30层: {orig_mb*30:.0f} MB ({orig_mb*30/1024:.2f} GB)")

    # 各 bit 压缩
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

        # 质量
        n_test = min(100, N)
        cos_list = []
        for i in range(n_test):
            v = flat[i].numpy()
            r = recon[i].numpy()
            cos = np.dot(v, r) / (np.linalg.norm(v) * np.linalg.norm(r) + 1e-10)
            cos_list.append(cos)

        log(f"\n  --- {bits}-bit TurboQuant ---")
        log(f"  单层: {orig_mb:.2f} MB → {comp_mb:.2f} MB (压缩比 {ratio:.1f}×)")
        log(f"  30层: {orig_mb*30/1024:.2f} GB → {comp_mb*30/1024:.2f} GB")
        log(f"  压缩: {ct:.3f}s | 解压: {dt:.3f}s")
        log(f"  余弦: 平均 {np.mean(cos_list):.4f}, 最低 {np.min(cos_list):.4f}")

    # 外推 720P/81帧
    tokens_720 = 21 * 45 * 80  # 75600
    log(f"\n  --- 外推 720P/81帧 (3-bit, 30层全缓存) ---")
    log(f"  实测 tokens: {N} → 720P tokens: {tokens_720} ({tokens_720/N:.0f}×)")
    fp32_720 = tokens_720 * d * 4 * 30
    bit3_720 = tokens_720 * (2 + d * 3 / 8) * 30 + 8 * 4 * 30
    log(f"  fp32:  {fp32_720/1024**3:.2f} GB")
    log(f"  3-bit: {bit3_720/1024**3:.2f} GB")
    log(f"  节省:  {(fp32_720-bit3_720)/1024**3:.2f} GB")
    log(f"  压缩质量: 余弦 {np.mean(cos_list):.4f} (实测)")

    # 清理显存
    del hidden, enc, captured, attn, flat, recon
    torch.mps.empty_cache() if hasattr(torch.mps, 'empty_cache') else None

log(f"\n{'='*70}")
log("  Benchmark 完成")
log(f"{'='*70}")
