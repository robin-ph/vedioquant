---
license: apache-2.0
base_model: Wan-AI/Wan2.2-TI2V-5B-Diffusers
tags:
  - video
  - video-generation
  - text-to-video
  - quantization
  - inference-optimization
  - diffusers
  - wan
  - wan2.2
language:
  - en
  - zh
library_name: diffusers
pipeline_tag: text-to-video
---

# Wan2.2-TI2V-5B-VedioQuant

**Wan2.2 + TurboQuant cache compression = 10x less VRAM for video generation**

This model packages [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) (the current best open-source video model family) with [VedioQuant](https://github.com/robin-ph/vedioquant) cache compression pre-configured. Just load and use — no extra setup needed.

## What's Different From Base Wan2.2?

|  | Base Wan2.2-5B | This Model (+ VedioQuant) |
|--|------------|---------------------------|
| Model weights | Unchanged | **Identical** (same weights) |
| Video quality | Baseline | **0.983 cosine similarity** (< 2% loss) |
| 720P/81f 30-layer cache | 25.96 GB | **2.44 GB** (10.6× smaller) |
| VRAM saved | — | **23.5 GB** |
| Extra code needed | N/A | **None** (auto-enabled) |

The model weights are **identical** to the original Wan2.2. VedioQuant only compresses the **inference cache** (not the weights) using TurboQuant's random rotation + quantization technique, originally developed for LLM KV Cache compression.

## Quick Start

```python
import torch
from pipeline_vedioquant import VedioQuantPipeline

# Load — VedioQuant is auto-enabled
pipe = VedioQuantPipeline.from_pretrained(
    "viberobin/Wan2.2-TI2V-5B-VedioQuant",
    torch_dtype=torch.float16,
)
pipe.to("cuda")  # or "mps" for Mac

# Generate video — same API as WanPipeline
video = pipe(
    prompt="a cat sitting on a sofa, cinematic lighting",
    num_frames=17,
    height=480,
    width=832,
).frames[0]

# Check compression stats
print(pipe.get_vedioquant_stats())
# → {'steps': 50, 'cache_hits': 32, 'hit_rate': '64%',
#    'bits': 3, 'compression_ratio': '10.7x'}
```

## Configuration

```python
# Adjust compression (default: 3-bit, recommended)
pipe.enable_vedioquant(bits=3, threshold=0.05)

# More aggressive compression (16x, slightly lower quality)
pipe.enable_vedioquant(bits=2)

# More conservative (8x, near-lossless)
pipe.enable_vedioquant(bits=4)

# Disable entirely (same as base Wan2.2)
pipe.disable_vedioquant()
```

## Benchmark (Real Measurement on Wan2.2-TI2V-5B)

Tested with real Wan2.2-5B attention features (hidden_dim=3072, 30 layers):

| Config | Compression | Cosine Sim | Min Cosine | Compress | Decompress |
|--------|------------|------------|------------|----------|------------|
| 2-bit | 16.0× | 0.9396 | 0.9363 | 53ms | 16ms |
| **3-bit** | **10.6×** | **0.9827** | **0.9811** | **14ms** | **14ms** |
| 4-bit | 8.0× | 0.9953 | 0.9946 | 14ms | 12ms |

### Gaussianization Verification

| Property | Before Rotation | After Rotation | Theory (Gaussian) |
|----------|----------------|----------------|-------------------|
| Kurtosis | 6.0 | **0.1** | 0.0 |
| Std dev | — | 0.018040 | 0.018042 |

Wan2.2 features are even more Gaussian-friendly than Wan2.1 (kurtosis 6 vs 15), resulting in excellent compression quality.

## Memory Savings (720P, 81 frames, 30-layer full cache)

| | fp32 | 3-bit VedioQuant | Saved |
|--|:---:|:---:|:---:|
| **Wan2.2-5B** | **25.96 GB** | **2.44 GB** | **23.52 GB** |
| Wan2.1-1.3B | 12.98 GB | 1.22 GB | 11.76 GB |

The 5B model has hidden_dim=3072 (vs 1536 for 1.3B), making cache compression even more impactful — saving **23.5 GB** of VRAM.

## How It Works

VedioQuant applies **TurboQuant** (Google Research, [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)) to compress the feature cache used by **TeaCache** ([arXiv:2411.19108](https://arxiv.org/abs/2411.19108)) during video diffusion inference:

1. **Random orthogonal rotation** transforms feature coordinates into a Gaussian distribution
2. **Precomputed Lloyd-Max codebook** quantizes each coordinate from 32 bits to 3 bits
3. Cache memory drops by **10.6×** with only **1.7% quality loss** (cosine similarity 0.983)

This is a **cross-domain transfer**: TurboQuant was designed for LLM KV Cache, but we discovered that video model attention features actually compress **better** than LLM features (kurtosis 6~15 vs 900).

## Citation

```bibtex
@misc{vedioquant2025,
  title={VedioQuant: Extreme Cache Compression for Video Diffusion Model Inference},
  author={Peng Han},
  year={2025},
  url={https://github.com/robin-ph/vedioquant}
}
```

## References

- TurboQuant: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) (Google Research)
- TeaCache: [arXiv:2411.19108](https://arxiv.org/abs/2411.19108) (CVPR 2025)
- Wan2.2: [GitHub](https://github.com/Wan-Video/Wan2.2) (Alibaba Cloud)
- VedioQuant: [GitHub](https://github.com/robin-ph/vedioquant)
