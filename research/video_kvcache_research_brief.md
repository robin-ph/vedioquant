# 视频模型推理降本研究：TurboQuant × TeaCache 融合实验

## 研究背景与目标

文生视频模型（如Wan2.1/2.2）推理成本极高，核心原因是扩散模型每次生成需要50步去噪，每步都要对全部视频token做完整的注意力计算（O(n²)复杂度）。

本研究探索将Google TurboQuant的KV向量极端压缩技术，迁移到视频模型的TeaCache加速框架中，降低缓存显存占用，进而提升可缓存步数和复用率。

---

## 关键概念说明

### 语言模型的KV Cache（原始场景）
自回归生成时，每生成一个token需要访问所有历史token的K、V向量。KV Cache将历史K/V存下来避免重复计算。序列越长，Cache越大。TurboQuant就是压缩这个Cache。

### 视频扩散模型的等价问题（迁移目标）
视频模型不是自回归，而是迭代去噪。但相邻去噪步之间，注意力层的中间特征变化极小。TeaCache利用这一点，缓存某些步的特征供下一步复用，跳过重复计算，实测加速2-3倍。

**迁移逻辑：**
- TeaCache缓存的特征向量 ≈ 语言模型的KV向量
- 用TurboQuant压缩这些缓存特征
- 显存减少 → 可缓存更多步 → 复用率更高 → 进一步加速

---

## TurboQuant技术原理

**论文**：TurboQuant (ICLR 2026, arXiv 2504.19874)  
**工程复现**：https://github.com/TheTom/turboquant_plus  
**Google官方博客**：https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/

### 核心算法（两阶段）

```
输入向量 x ∈ R^d（一个注意力head的K或V向量）
    │
    ├── 提取范数：γ = ||x||，x̂ = x/γ
    │
    ├── Stage 1: PolarQuant（b-1 bits）
    │   随机旋转 Π → 坐标变为 N(0, 1/d) 高斯分布
    │   → 对每个坐标做最优标量量化
    │
    ├── Stage 2: QJL（1 bit）
    │   sign(S · residual) → 无偏内积修正
    │
    └── 输出：CompressedVector(indices, signs, norms)
        总计：b bits/coordinate
```

### 已验证的压缩质量

| 配置 | 压缩比 | 余弦相似度 | MSE |
|------|--------|-----------|-----|
| 2-bit | 7.1× | 0.79 | 0.0047 |
| 3-bit | 4.9× | 0.91 | 0.0018 |
| 3.5-bit | 3.8× | **0.95** | 0.0009 |
| 4-bit | 3.8× | 0.96 | 0.0007 |

**3.5-bit是甜点：4倍压缩，0.95余弦相似度。**

### 关键验证指标（真实Qwen3 KV张量）
```
旋转前峰度：900.4  →  旋转后峰度：2.9（高斯分布=3.0）
旋转后标准差：0.088388
理论值(1/√d)：0.088388
比值：1.000（完全吻合）
```
旋转高斯化是整个算法有效的前提，已在语言模型KV张量上验证。**需要在视频模型特征上重新验证。**

---

## 实验设计：三阶段验证

### 第一阶段：TurboQuant基础验证（本地，无需GPU）

**目标**：跑通原始repo，理解压缩质量数字。

```bash
git clone https://github.com/TheTom/turboquant_plus
cd turboquant_plus
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 验证141个测试全过
python3 -m pytest tests/ -v

# 运行压缩demo
python3 benchmarks/demo.py
```

**预期输出**：各bit配置的余弦相似度表格，确认和README数字一致。

---

### 第二阶段：在视频模型特征上测压缩质量（云GPU，约2元）

**核心问题**：视频模型的注意力特征分布，和语言模型KV张量分布是否足够相似，使TurboQuant同样有效？

**步骤**：

1. 环境准备
```bash
pip install torch transformers diffusers
pip install git+https://github.com/Wan-Video/Wan2.1.git
```

2. Hook视频模型特征

```python
import torch
from diffusers import WanPipeline

pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

captured_features = []

def hook_fn(module, input, output):
    # 捕获注意力层的K、V向量
    captured_features.append(output.detach().cpu())

# 在第一个注意力块注册hook
hook = pipe.transformer.transformer_blocks[0].attn.register_forward_hook(hook_fn)

# 跑一次推理
with torch.no_grad():
    output = pipe(
        prompt="a cat sitting on a sofa",
        num_inference_steps=5,  # 少步数，只为拿特征
        height=480, width=480,
        num_frames=17,
    )

hook.remove()

# 保存特征
torch.save(captured_features, "video_features.pt")
print(f"捕获了 {len(captured_features)} 个特征张量")
print(f"特征形状：{captured_features[0].shape}")
```

3. 测量特征分布

```python
import torch
import numpy as np
from scipy.stats import kurtosis

features = torch.load("video_features.pt")
feat = features[0].float().numpy().flatten()

print(f"峰度(kurtosis): {kurtosis(feat):.1f}")  # 语言模型是900，越高越需要旋转
print(f"均值: {feat.mean():.4f}")
print(f"标准差: {feat.std():.4f}")

import matplotlib.pyplot as plt
plt.hist(feat[:10000], bins=100)
plt.title("Video model attention feature distribution")
plt.savefig("distribution.png")
```

4. 对视频特征跑TurboQuant压缩

```python
import sys
sys.path.insert(0, '/path/to/turboquant_plus')
from turboquant.turboquant import TurboQuant

# 取一个向量
vec = features[0][0, 0, :].float().numpy()  # shape: [d]
d = vec.shape[0]

tq = TurboQuant(dim=d, bits=3.5)

# 压缩
compressed = tq.compress(vec)

# 解压
reconstructed = tq.decompress(compressed)

# 计算质量
cosine_sim = np.dot(vec, reconstructed) / (np.linalg.norm(vec) * np.linalg.norm(reconstructed))
mse = np.mean((vec - reconstructed) ** 2)

print(f"余弦相似度: {cosine_sim:.4f}")
print(f"MSE: {mse:.6f}")
print(f"压缩比: {tq.compression_ratio:.1f}×")
```

**判断标准**：
- 余弦相似度 > 0.90 → 可以继续第三阶段
- 余弦相似度 0.80-0.90 → 需要调整bit数或添加针对视频特征的预处理
- 余弦相似度 < 0.80 → 分布差异太大，需要重新设计旋转矩阵

---

### 第三阶段：TeaCache + TurboQuant融合实现

**前提**：第二阶段余弦相似度 > 0.90。

**TeaCache原理**（Wan2.1已支持）：
```
去噪步 t=20 → 计算完整特征，存入cache
去噪步 t=21 → 特征变化小于阈值 → 直接用t=20的缓存，跳过计算
去噪步 t=22 → 变化超过阈值 → 重新计算，更新cache
```

**融合方案**：修改TeaCache的存储和读取逻辑

```python
class CompressedTeaCache:
    def __init__(self, bits=3.5, similarity_threshold=0.05):
        self.bits = bits
        self.threshold = similarity_threshold
        self.compressed_cache = {}
        self.tq_instances = {}  # 每个layer一个TurboQuant实例
    
    def should_recompute(self, layer_id, current_feat):
        """判断是否需要重新计算"""
        if layer_id not in self.compressed_cache:
            return True
        
        # 解压缓存特征做比较（只需要解压一小部分做快速判断）
        cached = self.decompress(layer_id)
        diff = torch.mean(torch.abs(current_feat - cached))
        return diff > self.threshold
    
    def store(self, layer_id, features):
        """压缩后存储"""
        d = features.shape[-1]
        if layer_id not in self.tq_instances:
            from turboquant.turboquant import TurboQuant
            self.tq_instances[layer_id] = TurboQuant(dim=d, bits=self.bits)
        
        tq = self.tq_instances[layer_id]
        # 批量压缩所有token的特征
        compressed = [tq.compress(features[i].numpy()) for i in range(len(features))]
        self.compressed_cache[layer_id] = compressed
    
    def decompress(self, layer_id):
        """解压取用"""
        tq = self.tq_instances[layer_id]
        compressed = self.compressed_cache[layer_id]
        return torch.stack([
            torch.tensor(tq.decompress(c)) for c in compressed
        ])
    
    def memory_usage(self):
        """统计显存占用"""
        total_bytes = 0
        for compressed_list in self.compressed_cache.values():
            for c in compressed_list:
                total_bytes += c.nbytes  # 压缩后的字节数
        return total_bytes / 1024**2  # MB
```

**对比实验设计**：

```python
configs = [
    {"name": "baseline", "teacache": False, "turboquant": False},
    {"name": "teacache_only", "teacache": True, "turboquant": False},
    {"name": "teacache+turbo3.5", "teacache": True, "turboquant": True, "bits": 3.5},
    {"name": "teacache+turbo3", "teacache": True, "turboquant": True, "bits": 3.0},
]

metrics = {
    "generation_time": [],    # 秒
    "peak_vram_mb": [],       # 显存峰值
    "cache_vram_mb": [],      # cache占用显存
    "video_quality_ssim": [], # 和baseline的SSIM
    "cosine_sim_avg": [],     # 缓存向量平均余弦相似度
}
```

---

## 代码结构建议

```
video_turboquant/
├── README.md
├── requirements.txt
├── phase1_verify_turboquant.py      # 第一阶段：基础验证
├── phase2_extract_features.py       # 第二阶段：提取视频特征
├── phase2_test_compression.py       # 第二阶段：测压缩质量
├── phase3_compressed_teacache.py    # 第三阶段：融合实现
├── phase3_benchmark.py              # 第三阶段：对比实验
└── results/
    ├── distribution_analysis.png
    ├── compression_quality.json
    └── benchmark_results.md
```

---

## 环境要求

```
Python >= 3.10
PyTorch >= 2.1
CUDA >= 11.8（或Apple Silicon MPS）
显存：
  - 第一阶段：无需GPU
  - 第二阶段：8GB（用Wan2.1-1.3B模型）
  - 第三阶段：16GB+（用Wan2.1-14B获得更有意义的结果）

依赖：
pip install torch diffusers transformers
pip install scipy numpy matplotlib
pip install git+https://github.com/TheTom/turboquant_plus.git
```

---

## 判断节点与决策树

```
第一阶段通过？
  └── YES → 进入第二阶段
  └── NO  → 检查turboquant_plus安装问题

第二阶段：余弦相似度多少？
  └── > 0.90 → 直接进入第三阶段
  └── 0.80-0.90 → 尝试4-bit，或研究视频特征分布特性再决定
  └── < 0.80 → 视频特征分布和语言模型差异太大，
               需要针对性设计旋转矩阵，这本身是一个研究贡献

第三阶段：融合后效果？
  └── 速度提升 + 显存减少 + 质量损失<5% → 成功，可以写技术报告
  └── 速度提升但质量损失>5% → 调整bits参数或threshold
  └── 速度没有提升 → TurboQuant解压开销抵消了复用收益，
                     需要优化解压实现（C扩展或GPU kernel）
```

---

## 参考资料

- TurboQuant论文：https://arxiv.org/abs/2504.19874
- TurboQuant工程复现：https://github.com/TheTom/turboquant_plus
- PolarQuant论文：https://arxiv.org/abs/2502.02617
- QJL论文：https://arxiv.org/abs/2406.03482
- Wan2.1 GitHub：https://github.com/Wan-Video/Wan2.1
- TeaCache论文：搜索"TeaCache video diffusion"
- Seaweed-7B技术报告：https://arxiv.org/html/2504.08685v1

---

## 研究价值说明

如果第三阶段验证成功，这个工作的贡献是：

1. 首次将语言模型KV Cache压缩技术迁移到视频扩散模型的特征缓存场景
2. 在TeaCache基础上实现显存进一步降低，使低显存设备可以运行更大模型
3. 为视频模型推理降本提供一个新的工程方向

成本估算：
- 第一阶段：0元（本地）
- 第二阶段：约2-5元（云GPU 1-2小时）
- 第三阶段：约10-20元（云GPU 5-10小时）

总计研究成本约20元人民币。
