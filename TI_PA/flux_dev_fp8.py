#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse, random, warnings
import torch
import torch.nn as nn
from PIL import Image
from diffusers import DiffusionPipeline

warnings.filterwarnings("ignore", category=FutureWarning)

SKIP_TYPES = (nn.Embedding, nn.LayerNorm, nn.ParameterList)

def safe_cast_fp8(module):
    ok = False
    for _, child in module.named_children():
        if isinstance(child, SKIP_TYPES):
            continue
        ok |= safe_cast_fp8(child)
    try:
        module.to(dtype=torch.float8_e4m3fn, device="cuda")
        ok = True
    except Exception:
        pass
    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="black-forest-labs/FLUX.1-dev")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", default="flux_test.png")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width",  type=int, default=512)
    ap.add_argument("--seed",   type=int, default=None)
    ap.add_argument("--fp8_strict", action="store_true")
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    token = os.getenv("HUGGINGFACE_HUB_TOKEN", None)

    print("[1/4] Loading pipeline...")
    pipe = DiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        use_safetensors=True,
        token=token,
    ).to("cuda")

    # 保持 scheduler 原样（Flux 自带）
    # pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)

    print("[2/4] Trying FP8 casting (UNet only)...")
    fp8_supported = hasattr(torch, "float8_e4m3fn")
    casted_any = False
    if fp8_supported:
        # 仅对 UNet 尝试 FP8
        unet = getattr(pipe, "unet", None)
        if unet is not None:
            casted_any = safe_cast_fp8(unet)
            print(f"   - unet FP8 cast success: {casted_any}")
    else:
        print("   - torch.float8_e4m3fn not available; skip FP8.")

    # text_encoder & VAE 保持 FP16
    pipe.text_encoder.to(dtype=torch.float16, device="cuda")
    pipe.vae.to(dtype=torch.float16, device="cuda")

    if not (fp8_supported and casted_any):
        if args.fp8_strict:
            raise RuntimeError("FP8 strict enabled but casting failed.")
        else:
            print("   >> FP8 not fully supported, fallback to FP16 for unsupported parts.")

    print("[3/4] Generating image...")
    with torch.autocast("cuda", dtype=torch.float16):
        result = pipe(
            args.prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            height=args.height,
            width=args.width,
        )

    img: Image.Image = result.images[0]
    img.save(args.out)
    print(f"[4/4] Done. Saved to {args.out}")

if __name__ == "__main__":
    main()

