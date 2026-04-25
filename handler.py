import os
import io
import base64
import tempfile
import threading
from pathlib import Path

import requests
import torch
import runpod
from diffusers import AutoencoderKLWan, WanAnimatePipeline
from diffusers.utils import load_image, load_video, export_to_video

MODEL_ID = os.getenv("WAN_MODEL_ID", "Wan-AI/Wan2.2-Animate-14B-Diffusers")
PIPE = None
LOCK = threading.Lock()

# Cache paths on attached RunPod network volume
os.environ.setdefault("HF_HOME", "/runpod-volume/hf")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/runpod-volume/hf/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/runpod-volume/hf/transformers")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

for p in [
    os.environ["HF_HOME"],
    os.environ["HUGGINGFACE_HUB_CACHE"],
    os.environ["TRANSFORMERS_CACHE"],
]:
    Path(p).mkdir(parents=True, exist_ok=True)


def _b64_to_file(b64_data: str, suffix: str) -> str:
    raw = base64.b64decode(b64_data)
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.write(raw)
    f.close()
    return f.name


def _url_to_file(url: str, suffix: str) -> str:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.write(r.content)
    f.close()
    return f.name


def _input_to_file(b64_value: str, url_value: str, suffix: str, label: str) -> str:
    if b64_value:
        return _b64_to_file(b64_value, suffix)
    if url_value:
        return _url_to_file(url_value, suffix)
    raise ValueError(f"Missing required input for {label}: provide base64 or URL")


def get_pipe():
    global PIPE
    if PIPE is None:
        with LOCK:
            if PIPE is None:
                hf_token = os.getenv("HF_TOKEN")
                if not hf_token:
                    raise RuntimeError("HF_TOKEN missing in endpoint env vars")

                dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

                vae = AutoencoderKLWan.from_pretrained(
                    MODEL_ID,
                    subfolder="vae",
                    torch_dtype=torch.float32,
                    token=hf_token,
                )

                PIPE = WanAnimatePipeline.from_pretrained(
                    MODEL_ID,
                    vae=vae,
                    torch_dtype=dtype,
                    token=hf_token,
                )

                if torch.cuda.is_available():
                    torch.backends.cuda.matmul.allow_tf32 = True
                    PIPE.to("cuda")

    return PIPE


def handler(job):
    data = job.get("input", {})

    prompt = data.get("prompt", "People in the video are doing actions.")
    mode = data.get("mode", "animate")  # animate | replacement

    image_b64 = data.get("image_base64")
    pose_b64 = data.get("pose_video_base64")
    face_b64 = data.get("face_video_base64")

    image_url = data.get("image_url")
    pose_url = data.get("pose_video_url")
    face_url = data.get("face_video_url")

    if not ((image_b64 or image_url) and (pose_b64 or pose_url) and (face_b64 or face_url)):
        return {
            "error": (
                "Wan2.2 Animate requires image + pose_video + face_video. "
                "Provide either *_base64 or *_url for each."
            )
        }

    seed = int(data.get("seed", 42))
    num_inference_steps = int(data.get("num_inference_steps", 20))
    guidance_scale = float(data.get("guidance_scale", 1.0))
    segment_frame_length = int(data.get("segment_frame_length", 77))
    prev_segment_conditioning_frames = int(data.get("prev_segment_conditioning_frames", 1))
    fps = int(data.get("fps", 24))

    image_path = pose_path = face_path = bg_path = mask_path = out_path = None
    try:
        image_path = _input_to_file(image_b64, image_url, ".png", "image")
        pose_path = _input_to_file(pose_b64, pose_url, ".mp4", "pose_video")
        face_path = _input_to_file(face_b64, face_url, ".mp4", "face_video")

        image = load_image(image_path)
        pose_video = load_video(pose_path)
        face_video = load_video(face_path)

        pipe = get_pipe()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(seed)

        kwargs = dict(
            image=image,
            pose_video=pose_video,
            face_video=face_video,
            prompt=prompt,
            mode=mode,
            segment_frame_length=segment_frame_length,
            prev_segment_conditioning_frames=prev_segment_conditioning_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

        if mode == "replacement":
            bg_b64 = data.get("background_video_base64")
            mask_b64 = data.get("mask_video_base64")
            bg_url = data.get("background_video_url")
            mask_url = data.get("mask_video_url")

            if not ((bg_b64 or bg_url) and (mask_b64 or mask_url)):
                return {
                    "error": (
                        "replacement mode requires background_video and mask_video "
                        "as base64 or URL."
                    )
                }

            bg_path = _input_to_file(bg_b64, bg_url, ".mp4", "background_video")
            mask_path = _input_to_file(mask_b64, mask_url, ".mp4", "mask_video")
            kwargs["background_video"] = load_video(bg_path)
            kwargs["mask_video"] = load_video(mask_path)

        frames = pipe(**kwargs).frames[0]

        out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        out_path = out_tmp.name
        out_tmp.close()
        export_to_video(frames, out_path, fps=fps)

        with open(out_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

        return {
            "video_base64": video_b64,
            "fps": fps,
            "seed": seed,
            "model": MODEL_ID,
        }

    except Exception as exc:
        return {"error": str(exc)}

    finally:
        for p in [image_path, pose_path, face_path, bg_path, mask_path, out_path]:
            if p and os.path.exists(p):
                os.remove(p)


runpod.serverless.start({"handler": handler})
