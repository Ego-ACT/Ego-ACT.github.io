
<div align="center">
<h2> EgoACT: Decoupling Action from Egocentric Observation for World Simulation</h2>

Yue Ma<sup>1</sup>, Pengjie Song<sup>1</sup>, Xinyu Wang<sup>2</sup>, Yi He<sup>2</sup>, Zeqian Long<sup>3</sup>, Fangneng Zhan<sup>1</sup>, Kaichen Zhou<sup>4</sup>, Hongyu Liu<sup>1</sup>, Hongfa Wang<sup>5</sup>, Peihao Li<sup>5</sup>, Haoyang Huang<sup>5</sup>, Nan Duan<sup>5</sup>, Qifeng Chen<sup>1&dagger;</sup>

<sup>1</sup>Hong Kong University of Science and Technology &nbsp; <sup>2</sup>Tsinghua University &nbsp; <sup>3</sup>Stanford University &nbsp; <sup>4</sup>Massachusetts Institute of Technology &nbsp; <sup>5</sup>JD Explore Academy

<sup>&dagger;</sup>Corresponding author


<a href='placeholder_arxiv_url'><img src='https://img.shields.io/badge/ArXiv-XXXX.XXXXX-red'></a>
<a href='https://jie-ser.github.io/EgoACT/'>
  <img src='https://img.shields.io/badge/Project-Page-Green'>
</a>
<!-- TODO: Replace with actual GitHub repo URL -->
[![GitHub](https://img.shields.io/github/stars/Jie-ser/EgoACT?style=social)](https://github.com/Jie-ser/EgoACT)

</div>

<!-- TODO: Add demo GIFs here -->
<!--
<table class="center">
  <td><img src="docs/gif_results/demo1.gif"></td>
  <td><img src="docs/gif_results/demo2.gif"></td>
  <tr>
  <td width=25% style="text-align:center;">"source → target"</td>
  <td width=25% style="text-align:center;">"source → target"</td>
</tr>
</table>
-->

## 🎏 Abstract

<b>TL; DR: <font color="red">EgoACT</font> decouples transferable action semantics from egocentric observations, enabling realistic action transfer for world simulation.</b>

<details><summary>CLICK for the full abstract</summary>

> Egocentric action transfer aims to decouple the semantic action from egocentric observations and reproduce them in new visual contexts, enabling scalable world simulation, embodied policy learning, and robotic data generation. Existing approaches to transferring actions rely on condition-guided video generation, which converts the observation video into explicit geometric controls (e.g., hand poses or meshes) to drive synthesis. However, these methods introduce geometric estimation errors, making it difficult to preserve physically plausible interactions when transferred to new scenes. Alternatively, motion transfer methods directly extract motion patterns from the observation video, yet motion-level features alone cannot encode rich contact dynamics, frequently leading to severe hand structural collapse and implausible contact layout. To address both limitations, we present EgoACT, a test-time framework for egocentric action transfer that operates directly on the denoising process of video diffusion models without requiring additional geometric estimators. EgoACT uses Velocity-guided Structure Anchoring to stabilize reference-consistent hand-object structure in the early denoising stage, and Sparse Correspondence Calibration to refine reliable local correspondences in the mid-to-late denoising stages. Together, these two components preserve transferable action semantics while improving temporal coherence and interaction realism. We further establish EgoActionBench, a benchmark for evaluating action preservation, visual quality, and hand-object plausibility across diverse egocentric manipulation scenarios. Experiments show that EgoACT generates more coherent and physically plausible interaction videos than strong baselines.

</details>


## 📋 Changelog

- 2026.10.07 Initial release

## 🛡 Setup Environment


```bash
# Create conda environment
conda create -n egoact python=3.10
conda activate egoact

# Install dependencies
cd EgoACT
pip install -e . # Editing mode
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 12.x
- 80GB+ GPU VRAM recommended (can run on lower VRAM with CPU offload)

## 📥 Model Download

We support two model variants:

| Model | VRAM Required | Quality |
|-------|---------------|---------|
| Wan2.1-T2V-1.3B | ~24GB | Good |
| Wan2.1-T2V-14B | ~80GB | Best |

### Using download script (Recommended)

```bash
# Download 14B model (default, best quality)
python examples/download_model.py --model 14b

# Download 1.3B model (lower VRAM requirement)
python examples/download_model.py --model 1.3b

# Download both models
python examples/download_model.py --model all
```

<details><summary>Or use ModelScope CLI directly:</summary>

```bash
# Download 14B model
modelscope download --model Wan-AI/Wan2.1-T2V-14B --local_dir ./models/Wan2.1-T2V-14B

# Download 1.3B model
modelscope download --model Wan-AI/Wan2.1-T2V-1.3B --local_dir ./models/Wan2.1-T2V-1.3B
```

</details>

## ⚔️ Inference

#### Quick Start

```bash
# Using 14B model with VSA + SCC (enabled by default)
python examples/wan_14b_text_to_video.py \
  --transfer_method egoact \
  --input_video data/source.mp4 \
  --prompt "your prompt here"

# Using 1.3B model (lower VRAM)
python examples/wan_1.3b_text_to_video.py \
  --transfer_method egoact \
  --input_video data/source.mp4 \
  --prompt "your prompt here"
```

#### Python API

```python
import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData

# Load models
model_manager = ModelManager(device="cpu")
model_manager.load_models([
    "models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
    "models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
    "models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
], torch_dtype=torch.bfloat16)

pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
pipe.enable_vram_management(num_persistent_param_in_dit=None)

# Load reference video for action transfer
ref_video = VideoData("data/source.mp4", height=480, width=832)

# Generate with action transfer (VSA + SCC enabled)
video = pipe(
    prompt="your prompt here",
    negative_prompt="low quality, blurry",
    num_inference_steps=50,
    denoising_strength=0.75,
    input_video=ref_video,
    seed=42,
    tiled=True,
    sf=4,
    transfer_method="egoact",
    vsa_enabled=True,
    scc_enabled=True,
)
save_video(video, "output.mp4", fps=15, quality=5)
```

#### Key Parameters

##### VSA (Velocity-guided Structure Anchoring)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--vsa_enabled` | `True` | Enable VSA for structural anchoring |
| `--vsa_optim_start` | `0` | First guidance step to apply VSA |
| `--vsa_optim_end` | `1` | Last guidance step to apply VSA |
| `--vsa_iter` | `2` | Optimization iterations per VSA step |
| `--vsa_mask_mode` | `uniform` | Spatial mask mode (`uniform`, `amf`) |

##### SCC (Sparse Correspondence Calibration)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--scc_enabled` | `True` | Enable SCC for correspondence calibration |
| `--scc_step_ratios` | `0.65 0.50 0.35 0.20` | Noise level ratios at which SCC activates |
| `--scc_topk` | `8` | Top-k local correspondences per token |
| `--scc_window_size` | `21` | Local search window size |
| `--scc_lr` | `0.0015` | Learning rate for pathwise correction |
| `--scc_stay_weight` | `0.05` | Stay regularizer weight |
| `--scc_anchor_mode` | `hybrid_corr` | Anchor mode (`ref_corr`, `hybrid_corr`, `residual_corr`) |

#### Transfer Methods

| Method | Description | Type |
|--------|-------------|------|
| `egoact` | AMF + VSA + SCC (default) | Training-free |
| `no_transfer` | Standard generation without motion transfer | Training-free |

## 📁 Project Structure

<details><summary>Click for directory structure</summary>

```
EgoACT/
├── diffsynth/                    # Core library
│   ├── models/                   # Model implementations
│   │   ├── wan_video_dit.py     # Modified DiT with Q/K extraction for AMF
│   │   ├── wan_video_vae.py     # Video VAE encoder/decoder
│   │   └── wan_video_text_encoder.py
│   ├── pipelines/               # Inference pipelines
│   │   └── wan_video.py         # Pipeline with AMF, VSA, and SCC
│   ├── schedulers/              # Noise schedulers (Flow Matching)
│   ├── prompters/               # Prompt processing
│   ├── benchmarks.py            # Benchmark presets and method taxonomy
│   └── vram_management/         # Memory optimization utilities
├── examples/                    # Example scripts
│   ├── wan_14b_text_to_video.py # CLI entry point (14B model)
│   ├── wan_1.3b_text_to_video.py # CLI entry point (1.3B model)
│   └── download_model.py       # Model downloader
├── docs/                        # GitHub Pages project page
├── models/                      # Model checkpoints (not tracked)
├── data/                        # Reference videos (not tracked)
├── requirements.txt            # Dependencies
└── setup.py                    # Package setup
```

</details>

## 📍 Citation

If you use this code, please cite:

```bibtex
@article{ma2026egoact,
  title={Decoupling Action from Egocentric Observation for World Simulation},
  author={Ma, Yue and Song, Pengjie and Wang, Xinyu and He, Yi and Long, Zeqian and Zhan, Fangneng and Zhou, Kaichen and Liu, Hongyu and Wang, Hongfa and Li, Peihao and Huang, Haoyang and Duan, Nan and Chen, Qifeng},
  journal={NeurIPS},
  year={2026}
}
```

<details><summary>Related work (FastVMT)</summary>

```bibtex
@article{ma2025fastvmt,
  title={FastVMT: Eliminating Redundancy in Video Motion Transfer},
  author={Ma, Yue and Wang, Zhikai and Ren, Tianhao and Zheng, Mingzhe and Liu, Hongyu and Guo, Jiayi and Feng, Kunyu and Xue, Yuxuan and Zhao, Zixiang and Schindler, Konrad and Chen, Qifeng and Zhang, Linfeng},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

</details>

## 📜 License

This project is open source and licensed under the MIT License. See [LICENSE.md](LICENSE.md) for details.

## 💗 Acknowledgements

This repository borrows heavily from [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) and [Wan Video](https://github.com/Wan-Video/Wan2.1). Thanks to the authors for sharing their code and models.

## 🧿 Maintenance

This is the codebase for our research work. If you have any questions or ideas to discuss, feel free to open an issue.
