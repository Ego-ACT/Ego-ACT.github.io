import argparse
import os
import re
import sys
from pathlib import Path

import torch

EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from diffsynth import (
    ModelManager,
    VideoData,
    WanVideoPipeline,
    save_video,
)
from diffsynth.benchmarks import (
    apply_benchmark_settings,
    build_run_metadata,
    get_method_family,
    normalize_transfer_method,
    write_metadata,
)


DEFAULT_MODEL_DIR = "models/Wan2.1-T2V-1.3B"
DEFAULT_DIT_MODEL = "diffusion_pytorch_model.safetensors"
DEFAULT_T5_MODEL = "models_t5_umt5-xxl-enc-bf16.pth"
DEFAULT_VAE_MODEL = "Wan2.1_VAE.pth"
DEFAULT_PROMPT = (
    "Documentary photography style. A lively puppy running quickly on a green grass field. "
    "The puppy has brown-yellow fur, ears perked up, with a focused and joyful expression. "
    "Sunlight shines on it, making the fur look extra soft and shiny. The background is an open "
    "grass field, occasionally dotted with wildflowers, with blue sky and white clouds visible in "
    "the distance. Strong perspective, capturing the puppy's dynamic movement and the vitality of "
    "the surrounding grass. Medium shot, side tracking view."
)
DEFAULT_NEGATIVE_PROMPT = (
    "vivid colors, overexposed, static, blurry details, subtitles, stylized, artwork, painting, "
    "still image, overall gray, worst quality, low quality, JPEG artifacts, ugly, incomplete, extra "
    "fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused "
    "fingers, static frame, cluttered background, three legs, many people in background, walking backwards"
)


def discover_model_paths(model_dir):
    return [
        os.path.join(model_dir, DEFAULT_DIT_MODEL),
        os.path.join(model_dir, DEFAULT_T5_MODEL),
        os.path.join(model_dir, DEFAULT_VAE_MODEL),
    ]


def get_model_paths(model_dir):
    return discover_model_paths(model_dir)


def get_next_video_path(output_dir="results", prefix="video", ext=".mp4"):
    os.makedirs(output_dir, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(ext)}$")
    nums = []
    for fn in os.listdir(output_dir):
        match = pattern.match(fn)
        if match:
            nums.append(int(match.group(1)))
    next_num = max(nums) + 1 if nums else 1
    return os.path.join(output_dir, f"{prefix}{next_num}{ext}")


def main(args):
    settings = apply_benchmark_settings(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        benchmark_preset=args.benchmark_preset,
    )

    model_manager = ModelManager(device="cpu")
    model_manager.load_models(discover_model_paths(args.model_dir), torch_dtype=torch.bfloat16)
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=None)

    input_video = VideoData(args.input_video, height=settings["height"], width=settings["width"])
    if args.benchmark_preset is not None:
        available_frames = len(input_video)
        if available_frames < settings["num_frames"]:
            raise ValueError(
                f"Reference video only has {available_frames} frames, but benchmark_preset "
                f"`{args.benchmark_preset}` requires {settings['num_frames']} frames."
            )
        input_video.set_length(settings["num_frames"])

    frames = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=settings["num_inference_steps"],
        denoising_strength=args.denoising_strength,
        input_video=input_video,
        seed=args.seed,
        tiled=True,
        height=settings["height"],
        width=settings["width"],
        num_frames=settings["num_frames"],
        sf=args.sf,
        test_latency=args.test_latency,
        latency_dir=args.latency_dir,
        transfer_method=args.transfer_method,
        benchmark_preset=args.benchmark_preset,
        mode=args.mode,
        scc_enabled=args.scc_enabled,
        scc_noise_levels=tuple(args.scc_noise_levels),
        scc_step_ratios=tuple(args.scc_step_ratios),
        scc_topk=args.scc_topk,
        scc_window_size=args.scc_window_size,
        scc_tau=args.scc_tau,
        scc_iter=args.scc_iter,
        scc_lr=args.scc_lr,
        scc_corr_weight=args.scc_corr_weight,
        scc_cycle_weight=args.scc_cycle_weight,
        scc_traj_weight=args.scc_traj_weight,
        scc_stay_weight=args.scc_stay_weight,
        scc_anchor_mode=args.scc_anchor_mode,
        scc_anchor_ref_weight=args.scc_anchor_ref_weight,
        scc_mask_mode=args.scc_mask_mode,
        scc_debug=args.scc_debug,
        guidance_steps=args.guidance_steps,
        vsa_enabled=args.vsa_enabled,
        vsa_optim_start=args.vsa_optim_start,
        vsa_optim_end=args.vsa_optim_end,
        vsa_iter=args.vsa_iter,
        vsa_scale_list=tuple(args.vsa_scale_list),
        vsa_mask_mode=args.vsa_mask_mode,
        vsa_mask_power=args.vsa_mask_power,
        vsa_mask_min=args.vsa_mask_min,
        vsa_balance_with_amf=args.vsa_balance_with_amf,
        vsa_debug=args.vsa_debug,
    )
    output_path = get_next_video_path(output_dir=args.output_dir)
    save_video(frames, output_path, fps=15, quality=5)

    summary = pipe.last_run_summary or {}
    resolved_method = summary.get("transfer_method")
    if resolved_method is None:
        resolved_method = normalize_transfer_method(args.transfer_method, args.mode)
    metadata = build_run_metadata(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        ref_video=args.input_video,
        output_path=output_path,
        seed=args.seed,
        steps=summary.get("num_inference_steps", settings["num_inference_steps"]),
        frames=summary.get("output_num_frames", summary.get("num_frames", len(frames))),
        height=summary.get("height", settings["height"]),
        width=summary.get("width", settings["width"]),
        method=resolved_method,
        model_variant=settings.get("model_variant", "Wan2.1-T2V-1.3B"),
        benchmark_preset=args.benchmark_preset,
        method_family=get_method_family(args.transfer_method, args.mode),
        extra={
            "requested_num_frames": summary.get("requested_num_frames", settings["num_frames"]),
            "decoded_num_frames": summary.get("decoded_num_frames", len(frames)),
            "output_num_frames": summary.get("output_num_frames", len(frames)),
        },
    )
    write_metadata(Path(output_path).with_suffix(".json"), metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WanVideo Text-to-Video Example")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--input_video", type=str, default="data/source.mp4")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sf", type=int, default=4)
    parser.add_argument("--test_latency", action="store_true")
    parser.add_argument("--latency_dir", type=str, default=None)
    parser.add_argument(
        "--transfer_method",
        type=str,
        default=None,
        choices=["egoact", "no_transfer"],
        help="Unified transfer method interface for Wan-native benchmark baselines",
    )
    parser.add_argument(
        "--benchmark_preset",
        type=str,
        default=None,
        choices=["wan14b_32f_832x480", "wan13b_32f_832x480"],
        help="Optional benchmark preset that overrides frames, resolution, and denoising steps",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="effi_AMF",
        choices=["No_transfer", "effi_AMF"],
        help="Legacy compatibility mode. Prefer --transfer_method for new benchmark runs.",
    )
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--guidance_steps", type=int, default=10)
    parser.add_argument("--vsa_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vsa_optim_start", type=int, default=0)
    parser.add_argument("--vsa_optim_end", type=int, default=1)
    parser.add_argument("--vsa_iter", type=int, default=2)
    parser.add_argument("--vsa_scale_list", type=float, nargs="+", default=[50.0, 300.0])
    parser.add_argument("--vsa_mask_mode", type=str, default="uniform", choices=["uniform", "amf"])
    parser.add_argument("--vsa_mask_power", type=float, default=1.0)
    parser.add_argument("--vsa_mask_min", type=float, default=0.15)
    parser.add_argument("--vsa_balance_with_amf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vsa_debug", action="store_true")
    parser.add_argument("--scc_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scc_noise_levels", type=float, nargs="+", default=[])
    parser.add_argument("--scc_step_ratios", type=float, nargs="+", default=[0.65, 0.50, 0.35, 0.20])
    parser.add_argument("--scc_topk", type=int, default=8)
    parser.add_argument("--scc_window_size", type=int, default=21)
    parser.add_argument("--scc_tau", type=float, default=1.0)
    parser.add_argument("--scc_iter", type=int, default=1)
    parser.add_argument("--scc_lr", type=float, default=0.0015)
    parser.add_argument("--scc_corr_weight", type=float, default=1.0)
    parser.add_argument("--scc_cycle_weight", type=float, default=0.0)
    parser.add_argument("--scc_traj_weight", type=float, default=0.0)
    parser.add_argument("--scc_stay_weight", type=float, default=0.05)
    parser.add_argument(
        "--scc_anchor_mode",
        type=str,
        default="hybrid_corr",
        choices=["ref_corr", "hybrid_corr", "residual_corr", "hybrid"],
    )
    parser.add_argument("--scc_anchor_ref_weight", type=float, default=0.25)
    parser.add_argument(
        "--scc_mask_mode",
        type=str,
        default="confidence",
        choices=["uniform", "none", "confidence", "confidence_first", "motion", "motion_aware", "motion_confidence"],
    )
    parser.add_argument("--scc_debug", action="store_true")
    parser.add_argument("--denoising_strength", type=float, default=0.75)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
