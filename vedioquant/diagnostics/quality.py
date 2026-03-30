"""
压缩质量诊断工具

在模型的真实特征上测量 TurboQuant 的压缩质量。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from ..compressor.turbo_quant import TurboQuantCompressor


def measure_quality(
    model: nn.Module,
    sample_inputs: Dict[str, torch.Tensor],
    bits: int = 3,
    num_vectors: int = 100,
    seed: int = 42,
) -> Dict:
    """
    在模型的真实特征上测量压缩质量。

    流程:
    1. Hook 注意力层，前向传播一次
    2. 捕获中间特征
    3. 对特征做 TurboQuant 压缩/解压
    4. 计算余弦相似度、MSE

    Args:
        model: transformer 模型
        sample_inputs: 模型 forward 的输入 kwargs
        bits: 量化位数
        num_vectors: 测试多少个向量
        seed: 随机种子

    Returns:
        {
            "per_layer": [{name, cosine_sim, mse, kurtosis_before, kurtosis_after}, ...],
            "average_cosine_sim": float,
            "average_mse": float,
            "bits": int,
            "compression_ratio": float,
        }
    """
    # 捕获注意力层输出
    captured = []
    hooks = []

    def make_hook(name):
        def fn(module, input, output):
            if len(captured) < 20:
                out = output[0] if isinstance(output, tuple) else output
                if isinstance(out, torch.Tensor):
                    captured.append({"name": name, "data": out.detach().cpu().float()})
        return fn

    for name, module in model.named_modules():
        if "attn" in name.lower() and "norm" not in name and "to_" not in name:
            if name.count(".") <= 2:
                hooks.append(module.register_forward_hook(make_hook(name)))
                if len(hooks) >= 10:
                    break

    # 前向传播
    with torch.no_grad():
        model(**sample_inputs)

    for h in hooks:
        h.remove()

    if not captured:
        return {"error": "未捕获到特征，请检查模型结构"}

    # 测量压缩质量
    results = []
    all_cos = []
    all_mse = []

    for feat_info in captured:
        data = feat_info["data"]
        flat = data.reshape(-1, data.shape[-1])
        d = flat.shape[-1]
        n = min(num_vectors, flat.shape[0])

        compressor = TurboQuantCompressor(dim=d, bits=bits, seed=seed)

        cos_list = []
        mse_list = []

        for j in range(n):
            vec = flat[j:j+1]
            compressed = compressor.compress(vec)
            recon = compressor.decompress(compressed)

            v = vec.flatten().numpy()
            r = recon.flatten().numpy()
            cos = np.dot(v, r) / (np.linalg.norm(v) * np.linalg.norm(r) + 1e-10)
            mse = np.mean((v - r) ** 2)
            cos_list.append(cos)
            mse_list.append(mse)

        # 旋转前后峰度
        from scipy.stats import kurtosis
        sample_vec = flat[0].numpy()
        gamma = np.linalg.norm(sample_vec)
        if gamma > 1e-10:
            x_hat = sample_vec / gamma
            Pi = compressor.polar.Pi.numpy()
            x_rot = Pi @ x_hat
            kurt_before = kurtosis(x_hat)
            kurt_after = kurtosis(x_rot)
        else:
            kurt_before = kurt_after = 0.0

        layer_result = {
            "name": feat_info["name"],
            "dim": d,
            "cosine_sim": float(np.mean(cos_list)),
            "cosine_sim_min": float(np.min(cos_list)),
            "mse": float(np.mean(mse_list)),
            "kurtosis_before_rotation": float(kurt_before),
            "kurtosis_after_rotation": float(kurt_after),
        }
        results.append(layer_result)
        all_cos.extend(cos_list)
        all_mse.extend(mse_list)

    return {
        "per_layer": results,
        "average_cosine_sim": float(np.mean(all_cos)),
        "average_mse": float(np.mean(all_mse)),
        "bits": bits,
        "compression_ratio": 32.0 / bits,
    }
