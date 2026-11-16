import asyncio
import base64
import hashlib
import io
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from dotenv import load_dotenv
from PIL import ExifTags, Image, ImageDraw, ImageFilter

AI_SOFTWARE_SIGNATURES = (
    "midjourney", "stable diffusion", "automatic1111", "comfyui", "dall-e",
    "dalle", "firefly", "adobe firefly", "bing image creator", "flux",
    "novelai", "leonardo", "invokeai", "c2pa", "synthid", "imagen",
)

CAMERA_MAKES = (
    "canon", "nikon", "sony", "fujifilm", "apple", "samsung", "google",
    "leica", "panasonic", "olympus", "hasselblad", "pentax", "ricoh",
)


def _to_data_url(img: Image.Image, fmt: str = "PNG", quality: int = 90) -> str:
    buf = io.BytesIO()
    if fmt.upper() == "JPEG":
        img.convert("RGB").save(buf, format="JPEG", quality=quality)
        mime = "image/jpeg"
    else:
        img.save(buf, format="PNG")
        mime = "image/png"
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _apply_colormap(norm_arr: np.ndarray, palette: str = "inferno") -> np.ndarray:
    """Maps a [0, 1] 2D float array to an RGB uint8 image without requiring matplotlib."""
    x = np.clip(norm_arr, 0.0, 1.0)
    if palette == "spectral_cyan":
        # Deep navy -> electric indigo -> cyan -> white for FFT spectrums
        r = np.clip(np.interp(x, [0.0, 0.35, 0.7, 1.0], [7, 65, 91, 245]), 0, 255)
        g = np.clip(np.interp(x, [0.0, 0.35, 0.7, 1.0], [10, 56, 192, 250]), 0, 255)
        b = np.clip(np.interp(x, [0.0, 0.35, 0.7, 1.0], [22, 165, 255, 255]), 0, 255)
    else:
        # Forensic thermal: dark navy -> violet -> crimson -> amber -> bright yellow
        r = np.clip(np.interp(x, [0.0, 0.25, 0.55, 0.8, 1.0], [10, 88, 215, 249, 255]), 0, 255)
        g = np.clip(np.interp(x, [0.0, 0.25, 0.55, 0.8, 1.0], [12, 28, 52, 148, 238]), 0, 255)
        b = np.clip(np.interp(x, [0.0, 0.25, 0.55, 0.8, 1.0], [30, 135, 95, 32, 130]), 0, 255)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def extract_metadata_forensics(img: Image.Image, raw_bytes: bytes) -> dict[str, Any]:
    exif_dict: dict[str, str] = {}
    try:
        raw_exif = img.getexif()
        if raw_exif:
            for tag_id, val in raw_exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                if isinstance(val, bytes):
                    try:
                        val = val.decode("utf-8", errors="ignore").strip("\x00")
                    except Exception:
                        continue
                exif_dict[str(tag_name)] = str(val)[:160]
    except Exception:
        pass

    info_text = " ".join(
        [str(v) for k, v in img.info.items() if isinstance(v, (str, bytes)) and len(str(v)) < 2000]
    ).lower()
    raw_head = raw_bytes[:16384].decode("latin-1", errors="ignore").lower()
    combined_meta = " ".join(exif_dict.values()).lower() + " " + info_text + " " + raw_head

    detected_ai_sig = next((s for s in AI_SOFTWARE_SIGNATURES if s in combined_meta), None)
    detected_camera = next((c for c in CAMERA_MAKES if c in combined_meta), None)

    has_quant_tables = hasattr(img, "quantization") and bool(getattr(img, "quantization", None))
    has_optical_exif = any(k in exif_dict for k in ("ExposureTime", "FNumber", "ISOSpeedRatings", "FocalLength"))

    score = 50.0  # neutral baseline
    signals = []
    if detected_ai_sig:
        score = 94.0
        signals.append(f"Embedded generator/C2PA signature detected: '{detected_ai_sig.upper()}'")
    elif detected_camera and has_optical_exif:
        score = 15.0
        signals.append(f"Authentic camera EXIF optics verified ({exif_dict.get('Make', detected_camera.title())} {exif_dict.get('Model', '')})")
    elif len(exif_dict) == 0:
        score = 62.0
        signals.append("Stripped EXIF container (common in synthetic exports & social media)")
    else:
        signals.append(f"Partial metadata headers ({len(exif_dict)} EXIF tags present)")

    return {
        "score": round(score, 1),
        "exif": exif_dict,
        "ai_signature": detected_ai_sig,
        "camera_signature": detected_camera,
        "has_optical_exif": has_optical_exif,
        "has_quantization": has_quant_tables,
        "signals": signals,
    }


def analyze_frequency_branch(rgb_arr: np.ndarray) -> dict[str, Any]:
    """2D Fast Fourier Transform (FFT) spectral analysis & periodic peak detector."""
    gray = np.dot(rgb_arr[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
    h, w = gray.shape
    # Windowing reduces border spectral leakage so true periodic peaks stand out
    win_y = np.hanning(h)[:, None]
    win_x = np.hanning(w)[None, :]
    windowed = (gray - np.mean(gray)) * (win_y * win_x)

    fft2 = np.fft.fftshift(np.fft.fft2(windowed))
    mag = np.log1p(np.abs(fft2))
    mag_norm = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)

    # Radial azimuthal profile and high-frequency periodic spike ratio
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.hypot(yy - cy, xx - cx)
    r_max = min(cy, cx)
    mid_high_mask = (r > r_max * 0.22) & (r < r_max * 0.88)

    region = mag_norm[mid_high_mask]
    mean_mh = float(np.mean(region))
    std_mh = float(np.std(region))
    p995 = float(np.percentile(region, 99.6))
    # Periodic upsampling grids (GAN/Diffusion transposed conv) create sharp off-center spikes
    spike_z = (p995 - mean_mh) / (std_mh + 1e-6)

    # High-to-mid frequency energy drop-off (diffusion often over-smooths or over-sharpens high bands)
    high_ring = mag_norm[(r >= r_max * 0.60) & (r < r_max * 0.92)]
    mid_ring = mag_norm[(r >= r_max * 0.18) & (r < r_max * 0.45)]
    hf_ratio = float(np.mean(high_ring) / (np.mean(mid_ring) + 1e-6))

    # Calibrate spectral anomaly score [0..100]
    spike_anomaly = np.clip((spike_z - 2.95) * 32.0, 0.0, 60.0)
    rolloff_anomaly = np.clip(abs(hf_ratio - 0.66) * 145.0, 0.0, 45.0)
    freq_score = float(np.clip(22.0 + spike_anomaly + rolloff_anomaly, 6.0, 97.0))

    fft_rgb = _apply_colormap(np.power(mag_norm, 0.85), palette="spectral_cyan")
    fft_img = Image.fromarray(fft_rgb)

    radial_bins = np.linspace(0, r_max, 32)
    radial_profile = []
    for i in range(len(radial_bins) - 1):
        ring_mask = (r >= radial_bins[i]) & (r < radial_bins[i + 1])
        radial_profile.append(round(float(np.mean(mag_norm[ring_mask])) if np.any(ring_mask) else 0.0, 3))

    return {
        "score": round(freq_score, 1),
        "spike_z": round(spike_z, 2),
        "hf_mid_ratio": round(hf_ratio, 3),
        "radial_profile": radial_profile,
        "fft_image": fft_img,
    }


def analyze_spatial_and_localization(img: Image.Image, rgb_arr: np.ndarray, has_optical_exif: bool = False) -> dict[str, Any]:
    """Computes Error Level Analysis (ELA), high-pass noise residuals, and suspicious-region localization heatmap."""
    h, w, _ = rgb_arr.shape

    # 1. Error Level Analysis (ELA) at Q=90
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    buf.seek(0)
    recompressed = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32)
    ela_diff = np.abs(rgb_arr.astype(np.float32) - recompressed)
    ela_gray = np.mean(ela_diff, axis=-1)
    ela_vis = np.clip(ela_diff * 14.0, 0, 255).astype(np.uint8)
    ela_img = Image.fromarray(ela_vis)

    # 2. High-pass SRM-style noise residual extraction
    gray = np.dot(rgb_arr[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
    blurred = np.asarray(
        Image.fromarray(gray.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=1.6)),
        dtype=np.float32,
    )
    residual = np.abs(gray - blurred)

    # 3. Block-wise variance map (16x16 grid) to detect localized inpainting / splicing vs global synthetic smoothness
    grid_h, grid_w = max(8, h // 16), max(8, w // 16)
    res_small = np.asarray(
        Image.fromarray(np.clip(residual * 6.0, 0, 255).astype(np.uint8)).resize((grid_w, grid_h), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    ela_small = np.asarray(
        Image.fromarray(np.clip(ela_gray * 10.0, 0, 255).astype(np.uint8)).resize((grid_w, grid_h), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )

    # Local deviation from median noise floor reveals spliced/inpainted patches;
    # abnormally uniform high-frequency correlation reveals full diffusion generation.
    res_med = np.median(res_small)
    res_mad = np.median(np.abs(res_small - res_med)) + 1e-4
    local_inconsistency = np.abs(res_small - res_med) / (res_mad + 2.5)

    combined_grid = 0.58 * (ela_small / (np.percentile(ela_small, 95) + 1e-4)) + 0.42 * (
        local_inconsistency / (np.percentile(local_inconsistency, 95) + 1e-4)
    )
    combined_grid = np.clip(combined_grid, 0.0, 1.25)
    combined_grid = (combined_grid - combined_grid.min()) / (combined_grid.max() - combined_grid.min() + 1e-6)

    # Upsample and smooth into a Grad-CAM style heatmap overlay
    heat_pil = Image.fromarray((combined_grid * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BICUBIC)
    heat_pil = heat_pil.filter(ImageFilter.GaussianBlur(radius=max(6, min(h, w) // 28)))
    heat_norm = np.asarray(heat_pil, dtype=np.float32) / 255.0

    heat_colored = _apply_colormap(heat_norm, palette="inferno").astype(np.float32)
    alpha = (0.22 + 0.56 * (heat_norm ** 1.35))[..., None]
    blended = np.clip(rgb_arr.astype(np.float32) * (1.0 - alpha) + heat_colored * alpha, 0, 255).astype(np.uint8)
    heatmap_img = Image.fromarray(blended)

    # Identify localized suspicious bounding regions from the coarse grid
    ela_p98 = float(np.percentile(ela_small, 98))
    ela_med = float(np.median(ela_small)) + 1e-3
    ela_ratio = ela_p98 / ela_med
    noise_inconsistency_p95 = float(np.percentile(local_inconsistency, 95))
    localization_ratio = max(ela_ratio, noise_inconsistency_p95 * 1.45)

    # Chroma-luminance micro-correlation & noise floor metrics
    noise_level = float(np.mean(residual))
    noise_kurtosis = float(np.mean(((residual - noise_level) / (np.std(residual) + 1e-5)) ** 4))

    is_localized_tamper = (
        not has_optical_exif
        and localization_ratio > 2.25
        and (float(np.std(ela_small)) > 2.0 or float(np.std(res_small)) > 3.2)
        and noise_level > 0.9
    )
    # Synthetic images often exhibit either ultra-low sensor grain or high kurtosis edge ringing
    synth_texture_score = 18.0 if has_optical_exif else 26.0
    if noise_level < 1.85:
        synth_texture_score += 36.0
    elif noise_level < 2.8 and not has_optical_exif:
        synth_texture_score += 14.0
    if noise_kurtosis > 14.0 and not has_optical_exif:
        synth_texture_score += 16.0
    if is_localized_tamper:
        synth_texture_score = max(synth_texture_score, min(88.0, 52.0 + localization_ratio * 7.5))

    spatial_score = float(np.clip(synth_texture_score, 8.0, 96.0))

    # Extract top 2 peak regions on a 4x4 macro grid for bounding box telemetry
    macro_h, macro_w = 4, 4
    macro_map = np.asarray(heat_pil.resize((macro_w, macro_h), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    flat_indices = np.argsort(macro_map.ravel())[::-1][:2]
    regions = []
    for idx in flat_indices:
        ry, rx = divmod(int(idx), macro_w)
        conf = float(macro_map[ry, rx])
        if conf > 0.55:
            regions.append({
                "x_pct": round(rx * 25.0 + 2.0, 1),
                "y_pct": round(ry * 25.0 + 2.0, 1),
                "w_pct": 21.0,
                "h_pct": 21.0,
                "intensity": round(conf * 100, 1),
                "reason": "Localized ELA & noise residual divergence" if is_localized_tamper else "High-frequency texture / edge coherence anomaly",
            })

    return {
        "score": round(spatial_score, 1),
        "is_localized_tamper": bool(is_localized_tamper),
        "localization_ratio": round(localization_ratio, 2),
        "noise_floor": round(noise_level, 2),
        "noise_kurtosis": round(noise_kurtosis, 2),
        "ela_image": ela_img,
        "heatmap_image": heatmap_img,
        "suspicious_regions": regions,
    }


async def run_vlm_branch(img: Image.Image, signal_summary: dict[str, Any]) -> dict[str, Any]:
    """Queries Groq (primary), OpenRouter (secondary free vision), or OpenAI (fallback) if configured in .env."""
    load_dotenv(override=True)
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not groq_key and not openrouter_key and not openai_key:
        return {
            "used": False,
            "provider": "Local Signal Forensics Only (No VLM Key in .env)",
            "model": "FFT + ELA + SRM Fusion",
            "score": None,
            "reasoning": "VLM API keys not configured in .env. Verdict computed from 2D FFT spectral peaks, ELA block residuals, noise kurtosis, and metadata headers.",
            "detected_artifacts": [],
            "manipulation_type": None,
            "suspected_family": None,
        }

    # Resize image to 384px for low-token, high-throughput VLM inspection
    thumb = img.copy().convert("RGB")
    thumb.thumbnail((384, 384), Image.Resampling.LANCZOS)
    data_url = _to_data_url(thumb, fmt="JPEG", quality=82)

    prompt = (
        "You are VisionForensics-X, an expert digital image forensics analyzer. "
        "Inspect this image alongside its signal telemetry: "
        f"FFT spike Z-score={signal_summary['fft_spike_z']}, "
        f"Noise floor={signal_summary['noise_floor']}, "
        f"ELA localization ratio={signal_summary['localization_ratio']}, "
        f"Localized tamper flag={signal_summary.get('is_localized_tamper', False)}, "
        f"Metadata note='{signal_summary['meta_note']}'.\n"
        "Evaluate physical lighting consistency, shadow geometry, reflections, anatomical/structural coherence, "
        "text/micro-texture rendering, and inpainting boundaries.\n"
        "Respond ONLY with a valid JSON object matching this exact schema:\n"
        "{\n"
        '  "synthetic_probability": <number 0-100>,\n'
        '  "manipulation_category": "Authentic Photograph" | "Full AI Synthesis" | "Localized Inpainting / Object Edit" | "Face Manipulation",\n'
        '  "suspected_generator": "Authentic Camera" | "Diffusion (Midjourney / Flux / SDXL)" | "GAN / Upsampler" | "Inpainting / Generative Fill",\n'
        '  "detected_artifacts": ["<short specific visual observation 1>", "<short specific visual observation 2>"],\n'
        '  "forensic_summary": "<concise 2-sentence technical explanation>"\n'
        "}"
    )

    providers = []
    if groq_key:
        groq_model = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b").strip()
        providers.append(("Groq VLM", "https://api.groq.com/openai/v1/chat/completions", groq_key, groq_model, True))
    if openrouter_key:
        or_model = os.getenv("OPENROUTER_VISION_MODEL", "nvidia/nemotron-3.5-super-vl:free").strip()
        providers.append(("OpenRouter VLM", "https://openrouter.ai/api/v1/chat/completions", openrouter_key, or_model, True))
        if or_model != "openrouter/free":
            providers.append(("OpenRouter Auto-Free VLM", "https://openrouter.ai/api/v1/chat/completions", openrouter_key, "openrouter/free", True))
    if groq_key:
        providers.append(("Groq Forensic Reasoning (GPT-OSS-120B)", "https://api.groq.com/openai/v1/chat/completions", groq_key, "openai/gpt-oss-120b", False))
    if openai_key:
        openai_model = os.getenv("OPENAI_API_KEY_MODEL") or os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini").strip()
        providers.append(("OpenAI Fallback", "https://api.openai.com/v1/chat/completions", openai_key, openai_model, True))

    last_err = None
    async with httpx.AsyncClient(timeout=25.0) as client:
        for provider_name, url, key, model_name, is_multimodal in providers:
            try:
                user_content: Any = (
                    [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ]
                    if is_multimodal
                    else prompt
                )
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": user_content}],
                    "temperature": 0.1,
                    "max_tokens": 500,
                }
                resp = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json=payload,
                )
                if resp.status_code == 429 and is_multimodal:
                    await asyncio.sleep(2.2)
                    resp = await client.post(
                        url,
                        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                        json=payload,
                    )
                if resp.status_code != 200:
                    last_err = f"{provider_name} ({model_name}) HTTP {resp.status_code}"
                    continue
                content = resp.json()["choices"][0]["message"]["content"].strip()
                start_idx = content.find("{")
                end_idx = content.rfind("}")
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    content = content[start_idx : end_idx + 1]
                parsed = json.loads(content)
                score = float(np.clip(float(parsed.get("synthetic_probability", 50.0)), 0.0, 100.0))
                return {
                    "used": True,
                    "provider": provider_name,
                    "model": model_name,
                    "score": round(score, 1),
                    "reasoning": parsed.get("forensic_summary", "VLM semantic analysis completed."),
                    "detected_artifacts": parsed.get("detected_artifacts", [])[:4],
                    "manipulation_type": parsed.get("manipulation_category"),
                    "suspected_family": parsed.get("suspected_generator"),
                }
            except Exception as exc:
                last_err = f"{provider_name}: {str(exc)[:120]}"
                continue

    return {
        "used": False,
        "provider": "Local Signal Fallback (VLM Unreachable)",
        "model": "FFT + ELA + SRM Fusion",
        "score": None,
        "reasoning": f"VLM request fell back to local signal engine ({last_err or 'API error'}).",
        "detected_artifacts": [],
        "manipulation_type": None,
        "suspected_family": None,
    }


async def analyze_image_pipeline(raw_bytes: bytes, filename: str = "upload.png") -> dict[str, Any]:
    t0 = time.perf_counter()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()

    img = Image.open(io.BytesIO(raw_bytes))
    orig_format = img.format or Path(filename).suffix.lstrip(".").upper() or "PNG"
    orig_w, orig_h = img.size

    # Standardize working resolution to keep FFT/ELA fast and consistent
    work_img = img.convert("RGB")
    if max(orig_w, orig_h) > 1024:
        work_img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    rgb_arr = np.asarray(work_img, dtype=np.uint8)

    meta_res = extract_metadata_forensics(img, raw_bytes)
    freq_res = analyze_frequency_branch(rgb_arr)
    spatial_res = analyze_spatial_and_localization(work_img, rgb_arr, has_optical_exif=meta_res["has_optical_exif"])

    vlm_res = await run_vlm_branch(
        work_img,
        {
            "fft_spike_z": freq_res["spike_z"],
            "noise_floor": spatial_res["noise_floor"],
            "localization_ratio": spatial_res["localization_ratio"],
            "is_localized_tamper": spatial_res["is_localized_tamper"],
            "meta_note": meta_res["signals"][0] if meta_res["signals"] else "None",
        },
    )

    # Feature Fusion Layer
    if meta_res["ai_signature"]:
        vlm_part = vlm_res["score"] if (vlm_res["used"] and vlm_res["score"] is not None) else 90.0
        fused_synth = max(89.0, 0.40 * meta_res["score"] + 0.25 * freq_res["score"] + 0.20 * spatial_res["score"] + 0.15 * vlm_part)
    elif spatial_res["is_localized_tamper"] and min(orig_w, orig_h) >= 380:
        vlm_part = vlm_res["score"] if (vlm_res["used"] and vlm_res["score"] is not None) else spatial_res["score"]
        fused_synth = max(62.0, 0.50 * spatial_res["score"] + 0.25 * freq_res["score"] + 0.25 * vlm_part)
    elif meta_res["has_optical_exif"] and not spatial_res["is_localized_tamper"]:
        vlm_part = vlm_res["score"] if (vlm_res["used"] and vlm_res["score"] is not None) else 18.0
        fused_synth = min(29.0, 0.35 * meta_res["score"] + 0.30 * spatial_res["score"] + 0.20 * freq_res["score"] + 0.15 * vlm_part)
    elif vlm_res["used"] and vlm_res["score"] is not None:
        fused_synth = (
            0.46 * vlm_res["score"]
            + 0.24 * spatial_res["score"]
            + 0.20 * freq_res["score"]
            + 0.10 * meta_res["score"]
        )
    else:
        fused_synth = (
            0.44 * spatial_res["score"]
            + 0.38 * freq_res["score"]
            + 0.18 * meta_res["score"]
        )

    fused_synth = float(np.clip(fused_synth, 2.0, 99.0))
    authenticity_score = round(100.0 - fused_synth, 1)

    # Multi-head classification: Verdict & Manipulation Category
    if min(orig_w, orig_h) < 380 and not meta_res["ai_signature"]:
        verdict = "Uncertain / Borderline"
        manipulation_category = "Heavy Recompression / Degraded Resolution"
    elif spatial_res["is_localized_tamper"] and fused_synth >= 46.0:
        verdict = "AI-Edited / Manipulated"
        manipulation_category = "Localized Inpainting / Region Splicing"
    elif fused_synth >= 58.0:
        verdict = "AI-Generated"
        manipulation_category = vlm_res.get("manipulation_type") or "Full Synthetic Generation"
    elif fused_synth <= 42.0:
        verdict = "Authentic Photograph"
        manipulation_category = "Unaltered Optical Capture"
    else:
        verdict = "Uncertain / Borderline"
        manipulation_category = "Ambiguous Compression / Post-Processed"

    # Family Attribution probabilities
    if verdict == "Authentic Photograph":
        attribution = [
            {"family": "Authentic Optical Sensor", "probability": round(max(65.0, authenticity_score), 1)},
            {"family": "Diffusion (Flux / SDXL / Midjourney)", "probability": round((100 - authenticity_score) * 0.55, 1)},
            {"family": "Inpainting / Region Edit", "probability": round((100 - authenticity_score) * 0.45, 1)},
        ]
    elif verdict == "AI-Edited / Manipulated":
        attribution = [
            {"family": "Inpainting / Generative Fill", "probability": round(min(91.0, fused_synth + 10.0), 1)},
            {"family": "Diffusion Full Synthesis", "probability": round(max(6.0, (100 - fused_synth) * 0.5), 1)},
            {"family": "Authentic Base + Local Splice", "probability": round(max(12.0, 100 - fused_synth), 1)},
        ]
    else:
        diff_prob = round(min(94.0, fused_synth * (0.82 if freq_res["spike_z"] < 4.2 else 0.62)), 1)
        gan_prob = round(max(4.0, fused_synth - diff_prob), 1)
        attribution = [
            {"family": vlm_res.get("suspected_family") or "Diffusion (Midjourney / Flux / SDXL)", "probability": diff_prob},
            {"family": "GAN / Neural Upsampler", "probability": gan_prob},
            {"family": "Authentic Optical Sensor", "probability": round(max(2.0, 100.0 - fused_synth), 1)},
        ]

    # Uncertainty & Reliability Engine
    uncertainty_reasons = []
    if min(orig_w, orig_h) < 380:
        uncertainty_reasons.append(f"Low input resolution ({orig_w}x{orig_h}px) attenuates high-frequency FFT peaks.")
    if not vlm_res["used"]:
        uncertainty_reasons.append("Running in offline signal-only mode (add GROQ_API_KEY in .env for VLM semantic fusion).")
    if abs(spatial_res["score"] - freq_res["score"]) > 34.0:
        uncertainty_reasons.append("Spatial and frequency branches show divergent confidence (possible heavy JPEG recompression).")

    if len(uncertainty_reasons) == 0 and (fused_synth > 74 or fused_synth < 26):
        uncertainty_level = "Low (High Reliability)"
        reliability_pct = 92
    elif len(uncertainty_reasons) <= 1:
        uncertainty_level = "Moderate"
        reliability_pct = 78
    else:
        uncertainty_level = "Elevated (Degraded Input)"
        reliability_pct = 61

    findings = list(meta_res["signals"])
    findings.append(
        f"2D FFT spectral peak Z-score: {freq_res['spike_z']} (HF/Mid energy ratio: {freq_res['hf_mid_ratio']})"
    )
    findings.append(
        f"Spatial noise floor: {spatial_res['noise_floor']} dB | ELA localization ratio: {spatial_res['localization_ratio']}x"
    )
    if vlm_res["detected_artifacts"]:
        findings.extend(vlm_res["detected_artifacts"])

    elapsed_ms = round((time.perf_counter() - t0) * 1000)

    return {
        "report_id": f"VFX-{sha256[:8].upper()}",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "filename": filename,
        "sha256": sha256[:24] + "…",
        "dimensions": f"{orig_w} × {orig_h}",
        "format": orig_format,
        "file_size_kb": round(len(raw_bytes) / 1024, 1),
        "elapsed_ms": elapsed_ms,
        "verdict": verdict,
        "synthetic_probability": round(fused_synth, 1),
        "authenticity_score": authenticity_score,
        "manipulation_category": manipulation_category,
        "uncertainty": {
            "level": uncertainty_level,
            "reliability_pct": reliability_pct,
            "reasons": uncertainty_reasons,
        },
        "branch_scores": {
            "spatial": spatial_res["score"],
            "frequency": freq_res["score"],
            "metadata": meta_res["score"],
            "vlm": vlm_res["score"] if vlm_res["score"] is not None else round(fused_synth, 1),
            "vlm_active": vlm_res["used"],
            "vlm_provider": vlm_res["provider"],
            "vlm_model": vlm_res["model"],
        },
        "vlm_reasoning": vlm_res["reasoning"],
        "attribution": attribution,
        "findings": findings,
        "suspicious_regions": spatial_res["suspicious_regions"],
        "radial_spectrum": freq_res["radial_profile"],
        "metadata_tags": meta_res["exif"],
        "visual_layers": {
            "original": _to_data_url(work_img, fmt="JPEG", quality=90),
            "heatmap": _to_data_url(spatial_res["heatmap_image"], fmt="JPEG", quality=90),
            "fft": _to_data_url(freq_res["fft_image"], fmt="PNG"),
            "ela": _to_data_url(spatial_res["ela_image"], fmt="JPEG", quality=90),
        },
    }


def ensure_sample_images(samples_dir: Path) -> list[dict[str, str]]:
    """Generates 10 deterministic forensic test images across Authentic, AI Diffusion, Inpainted/Edited, and Degraded categories."""
    samples_dir.mkdir(parents=True, exist_ok=True)
    from PIL.PngImagePlugin import PngInfo
    rng = np.random.default_rng(42)
    h, w = 480, 640
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    def _save_camera_jpeg(path: Path, arr: np.ndarray, make: str, model: str, iso: int, fnum: tuple[int, int], picsum_id: str = ""):
        if path.exists():
            return
        pil_img = None
        if picsum_id:
            try:
                resp = httpx.get(f"https://picsum.photos/id/{picsum_id}/640/480.jpg", follow_redirects=True, timeout=6.0)
                if resp.status_code == 200:
                    pil_img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            except Exception:
                pil_img = None
        if pil_img is None:
            grain = rng.normal(0, 4.1, size=(h, w, 3))
            noisy = np.clip(arr + grain, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(noisy)
        exif = pil_img.getexif()
        exif[271] = make
        exif[272] = model
        exif[305] = f"{make} Optical Pipeline"
        exif[33434] = (1, 250)
        exif[33437] = fnum
        exif[34855] = iso
        pil_img.save(path, format="JPEG", quality=96, exif=exif.tobytes())

    def _save_ai_png(path: Path, base: np.ndarray, software: str, prompt_meta: str, freq_div: float = 4.0):
        if path.exists():
            return
        grid_artifact = 4.8 * (np.sin(xx * np.pi / freq_div) * np.sin(yy * np.pi / freq_div))[..., None]
        synth_arr = np.clip(base + grid_artifact, 0, 255).astype(np.uint8)
        pil_ai = Image.fromarray(synth_arr)
        meta = PngInfo()
        meta.add_text("Software", software)
        meta.add_text("parameters", prompt_meta)
        pil_ai.save(path, format="PNG", pnginfo=meta)

    def _save_spliced_jpeg(path: Path, base_path: Path, box: tuple[int, int, int, int], colors: tuple[float, float, float]):
        if path.exists():
            return
        base_img = Image.open(base_path).convert("RGB")
        buf = io.BytesIO()
        base_img.save(buf, format="JPEG", quality=82)
        buf.seek(0)
        comp_arr = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32).copy()
        y0, y1, x0, x1 = box
        py, px = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        pr = colors[0] + 38 * np.sin(px / 7.0)
        pg = colors[1] + 50 * np.cos(py / 7.0)
        pb = colors[2] + 55 * np.sin((px + py) / 9.0)
        comp_arr[y0:y1, x0:x1] = np.stack([pr, pg, pb], axis=-1)
        Image.fromarray(np.clip(comp_arr, 0, 255).astype(np.uint8)).save(path, format="JPEG", quality=98)

    # 1. Authentic Camera: Canon EOS R5 Alpine Horizon
    p1 = samples_dir / "authentic_camera_capture.jpg"
    sky1 = np.stack([38 + 75 * (yy / h), 68 + 95 * (yy / h), 125 + 80 * (1.0 - yy / h)], axis=-1)
    r1 = 230 + 38 * np.sin(xx / 52.0) + 18 * np.cos(xx / 19.0)
    r2 = 305 + 28 * np.cos(xx / 37.0 + 1.2) + 12 * np.sin(xx / 13.0)
    arr1 = sky1.copy()
    arr1[yy > r1] = np.stack([42, 55, 78], axis=-1)
    arr1[yy > r2] = np.stack([26, 48, 42], axis=-1)
    _save_camera_jpeg(p1, arr1, "Canon", "Canon EOS R5", 160, (28, 10), picsum_id="29")

    # 2. AI Diffusion: SDXL / ComfyUI Neon Portrait
    p2 = samples_dir / "ai_diffusion_synth.png"
    rad2 = np.hypot(xx - w * 0.5, yy - h * 0.46)
    base2 = np.stack([
        24 + 165 * np.exp(-rad2 / 180.0) + 22 * np.sin(xx / 28.0),
        18 + 95 * np.exp(-rad2 / 150.0) + 35 * np.cos(yy / 32.0),
        55 + 185 * np.exp(-rad2 / 220.0),
    ], axis=-1)
    _save_ai_png(p2, base2, "Stable Diffusion XL / ComfyUI", "cyberpunk neon portrait, 8k, DPM++ 2M Karras")

    # 3. Manipulated / Inpainted: Regional Object Splice
    p3 = samples_dir / "inpainted_region_edit.jpg"
    _save_spliced_jpeg(p3, p1, (int(h * 0.28), int(h * 0.68), int(w * 0.52), int(w * 0.84)), (210, 75, 185))

    # 4. Degraded / Recompressed Social Media Image
    p4 = samples_dir / "recompressed_social_media.jpg"
    if not p4.exists():
        small = Image.open(p1).convert("RGB").resize((300, 220), Image.Resampling.BILINEAR)
        small.save(p4, format="JPEG", quality=42)

    # 5. Authentic Camera: Sony A7 IV Coastal Sunset
    p5 = samples_dir / "sony_a7iv_coastal_sunset.jpg"
    arr5 = np.stack([
        180 * (1.0 - yy / h) + 45,
        95 * (1.0 - yy / h) + 55 + 20 * np.sin(xx / 40.0),
        70 + 110 * (yy / h),
    ], axis=-1)
    arr5[yy > h * 0.62] = np.stack([22, 58, 92], axis=-1) + 14 * np.sin(yy[yy > h * 0.62, None] / 4.0)
    _save_camera_jpeg(p5, arr5, "Sony", "Sony ILCE-7M4", 200, (40, 10), picsum_id="15")

    # 6. AI Synthesis: Midjourney v6.1 Hyperreal Portrait
    p6 = samples_dir / "midjourney_v6_portrait.png"
    rad6 = np.hypot(xx - w * 0.48, yy - h * 0.42)
    base6 = np.stack([
        40 + 170 * np.exp(-rad6 / 160.0),
        32 + 135 * np.exp(-rad6 / 145.0) + 18 * np.sin(xx / 24.0),
        48 + 115 * np.exp(-rad6 / 190.0),
    ], axis=-1)
    _save_ai_png(p6, base6, "Midjourney v6.1", "editorial studio portrait, cinematic rim lighting --v 6.1 --stylize 250", freq_div=3.5)

    # 7. AI Synthesis: Flux.1-Dev Architectural Render
    p7 = samples_dir / "flux1_dev_architecture.png"
    base7 = np.stack([
        35 + 140 * (xx / w) + 28 * np.cos(xx / 18.0),
        48 + 150 * (1.0 - yy / h) + 22 * np.sin(yy / 20.0),
        85 + 130 * (xx / w),
    ], axis=-1)
    _save_ai_png(p7, base7, "Flux.1-dev", "brutalist glass museum atrium, volumetric sunlight, octane render", freq_div=4.0)

    # 8. Manipulated: Generative Fill Face / Center Swap
    p8 = samples_dir / "generative_fill_face_edit.jpg"
    _save_spliced_jpeg(p8, p5, (int(h * 0.22), int(h * 0.64), int(w * 0.34), int(w * 0.66)), (85, 195, 220))

    # 9. Authentic Camera: Fujifilm X-T5 Forest Canopy
    p9 = samples_dir / "fujifilm_xt5_forest.jpg"
    arr9 = np.stack([
        28 + 35 * np.sin(xx / 23.0) + 20 * (yy / h),
        68 + 55 * np.cos(yy / 29.0) + 30 * (1.0 - yy / h),
        36 + 28 * np.sin((xx + yy) / 31.0),
    ], axis=-1)
    _save_camera_jpeg(p9, arr9, "Fujifilm", "Fujifilm X-T5", 125, (20, 10), picsum_id="28")

    # 10. AI Synthesis: DALL-E 3 C2PA Concept Art
    p10 = samples_dir / "dalle3_c2pa_concept.png"
    rad10 = np.hypot(xx - w * 0.55, yy - h * 0.52)
    base10 = np.stack([
        65 + 145 * np.sin(rad10 / 42.0) ** 2,
        30 + 160 * np.exp(-rad10 / 210.0),
        90 + 140 * np.cos(rad10 / 55.0) ** 2,
    ], axis=-1)
    _save_ai_png(p10, base10, "DALL-E 3 / C2PA SynthID", "surreal bioluminescent botanical sphere, studio macro", freq_div=4.5)

    return [
        {"id": "authentic_camera_capture.jpg", "title": "01 · Canon EOS R5 Alpine", "tag": "Authentic Camera", "badge": "real", "url": "/samples/authentic_camera_capture.jpg"},
        {"id": "ai_diffusion_synth.png", "title": "02 · SDXL ComfyUI Neon", "tag": "AI Generated", "badge": "ai", "url": "/samples/ai_diffusion_synth.png"},
        {"id": "inpainted_region_edit.jpg", "title": "03 · Regional Object Splice", "tag": "Manipulated", "badge": "edited", "url": "/samples/inpainted_region_edit.jpg"},
        {"id": "recompressed_social_media.jpg", "title": "04 · WhatsApp Re-encode", "tag": "Low-Res / Uncertain", "badge": "uncertain", "url": "/samples/recompressed_social_media.jpg"},
        {"id": "sony_a7iv_coastal_sunset.jpg", "title": "05 · Sony A7 IV Sunset", "tag": "Authentic Camera", "badge": "real", "url": "/samples/sony_a7iv_coastal_sunset.jpg"},
        {"id": "midjourney_v6_portrait.png", "title": "06 · Midjourney v6.1 Studio", "tag": "AI Generated", "badge": "ai", "url": "/samples/midjourney_v6_portrait.png"},
        {"id": "flux1_dev_architecture.png", "title": "07 · Flux.1-Dev Atrium", "tag": "AI Generated", "badge": "ai", "url": "/samples/flux1_dev_architecture.png"},
        {"id": "generative_fill_face_edit.jpg", "title": "08 · Generative Fill Edit", "tag": "Manipulated", "badge": "edited", "url": "/samples/generative_fill_face_edit.jpg"},
        {"id": "fujifilm_xt5_forest.jpg", "title": "09 · Fujifilm X-T5 Foliage", "tag": "Authentic Camera", "badge": "real", "url": "/samples/fujifilm_xt5_forest.jpg"},
        {"id": "dalle3_c2pa_concept.png", "title": "10 · DALL-E 3 C2PA Concept", "tag": "AI Generated", "badge": "ai", "url": "/samples/dalle3_c2pa_concept.png"},
    ]


def _account_paths(base_dir: Path, account: str = "mini") -> tuple[Path, Path]:
    safe = "".join(c for c in account.lower() if c.isalnum() or c in ("_", "-")) or "mini"
    repo_path = base_dir / "data" / "accounts" / f"{safe}.json"
    tmp_path = Path("/tmp/vfx_accounts") / f"{safe}.json"
    return repo_path, tmp_path


def load_account_history(base_dir: Path, account: str = "mini") -> list[dict[str, Any]]:
    repo_path, tmp_path = _account_paths(base_dir, account)
    target = tmp_path if tmp_path.exists() else repo_path
    if not target.exists():
        return []
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        return data.get("history", [])
    except Exception:
        return []


def save_to_account_history(base_dir: Path, report: dict[str, Any], account: str = "mini") -> list[dict[str, Any]]:
    history = load_account_history(base_dir, account)
    history = [h for h in history if not (h.get("report_id") == report.get("report_id") and h.get("filename") == report.get("filename"))]
    history.insert(0, report)
    history = history[:50]
    payload = json.dumps({"account": account, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), "history": history}, indent=2)
    repo_path, tmp_path = _account_paths(base_dir, account)
    try:
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        repo_path.write_text(payload, encoding="utf-8")
    except OSError:
        # Vercel serverless functions have a read-only bundle; persist warm-session updates in /tmp
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(payload, encoding="utf-8")
    return history


def clear_account_history(base_dir: Path, account: str = "mini") -> None:
    payload = json.dumps({"account": account, "history": []}, indent=2)
    repo_path, tmp_path = _account_paths(base_dir, account)
    try:
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        repo_path.write_text(payload, encoding="utf-8")
    except OSError:
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(payload, encoding="utf-8")
