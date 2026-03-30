"""端到端测试：在 Wan2.1 上验证 vedioquant 完整流程"""

import torch
import vedioquant


def test_wan21_full_flow():
    """完整流程：加载模型 → enable → 推理 → stats → disable"""
    from diffusers.models import WanTransformer3DModel

    # 加载模型
    model = WanTransformer3DModel.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        subfolder="transformer",
        torch_dtype=torch.float32,
    )
    model = model.to("mps")
    model.eval()

    # 启用压缩缓存
    handle = vedioquant.enable(model, bits=3, threshold=0.15)

    # 模拟多步去噪
    torch.manual_seed(42)
    encoder_hidden = torch.randn(1, 32, 4096, dtype=torch.float32, device="mps")
    base = torch.randn(1, 16, 1, 16, 16, dtype=torch.float32, device="mps")

    for step in range(10):
        t = 1000 - step * 100
        hidden = base + torch.randn_like(base) * 0.005
        with torch.no_grad():
            out = model(
                hidden_states=hidden,
                encoder_hidden_states=encoder_hidden,
                timestep=torch.tensor([float(t)], device="mps"),
            )
        base = hidden

    # 检查统计
    stats = handle.stats()
    print(f"Stats: {stats}")
    assert stats["steps"] == 10
    assert stats["bits"] == 3
    assert "hit_rate" in stats

    # 禁用
    vedioquant.disable(handle)
    print("端到端测试通过!")


def test_estimate_savings():
    """显存估算功能"""
    result = vedioquant.estimate_savings(
        hidden_dim=1536, num_layers=30,
        height=720, width=1280, num_frames=81,
        bits=3, cache_layers=2,
    )
    print(f"720P 81帧: fp32={result['fp32_cache']}, 3bit={result['compressed_cache']}, 节省={result['saved']}")

    assert result["compression_ratio"] > 5
    assert result["saved_bytes"] > 0
    assert result["tokens"] > 0


def test_diagnose():
    """诊断功能"""
    from diffusers.models import WanTransformer3DModel

    model = WanTransformer3DModel.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        subfolder="transformer",
        torch_dtype=torch.float32,
    )
    model = model.to("mps")
    model.eval()

    report = vedioquant.diagnose(
        model,
        sample_inputs={
            "hidden_states": torch.randn(1, 16, 1, 16, 16, dtype=torch.float32, device="mps"),
            "encoder_hidden_states": torch.randn(1, 32, 4096, dtype=torch.float32, device="mps"),
            "timestep": torch.tensor([500.0], dtype=torch.float32, device="mps"),
        },
        bits=3,
    )
    print(f"诊断结果: 平均余弦={report['average_cosine_sim']:.4f}, 压缩比={report['compression_ratio']:.1f}×")

    assert report["average_cosine_sim"] > 0.90
    assert report["compression_ratio"] > 5


if __name__ == "__main__":
    test_estimate_savings()
    test_wan21_full_flow()
    test_diagnose()
    print("\n所有端到端测试通过!")
