import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.staticfiles import StaticFiles

from forensics import (
    analyze_image_pipeline,
    clear_account_history,
    ensure_sample_images,
    load_account_history,
    save_to_account_history,
)

BASE_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = BASE_DIR / "samples"
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI(title="VisionForensics-X", version="1.0.0")

SAMPLE_MANIFEST = ensure_sample_images(SAMPLES_DIR)

MAX_IMAGE_BYTES = 12 * 1024 * 1024  # 12 MB


@app.get("/api/status")
def get_engine_status():
    load_dotenv(override=True)
    groq_configured = bool(os.getenv("GROQ_API_KEY", "").strip())
    openrouter_configured = bool(os.getenv("OPENROUTER_API_KEY", "").strip())
    openai_configured = bool(os.getenv("OPENAI_API_KEY", "").strip())
    active_providers = []
    if groq_configured:
        active_providers.append("Groq")
    if openrouter_configured:
        active_providers.append("OpenRouter")
    if openai_configured:
        active_providers.append("OpenAI")
    return {
        "engine": "VisionForensics-X Dual-Branch + VLM",
        "groq_configured": groq_configured,
        "openrouter_configured": openrouter_configured,
        "openai_configured": openai_configured,
        "active_mode": (
            f"{' + '.join(active_providers)} VLM + Signal Fusion"
            if active_providers
            else "Local Signal Forensics (Add keys in .env)"
        ),
        "groq_model": os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b"),
        "openrouter_model": os.getenv("OPENROUTER_VISION_MODEL", "nvidia/nemotron-3.5-super-vl:free"),
        "openai_model": os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini"),
    }


@app.get("/api/samples")
def list_samples():
    return {"samples": ensure_sample_images(SAMPLES_DIR)}


@app.get("/api/account/{account}/history")
def get_account_history(account: str = "mini"):
    history = load_account_history(BASE_DIR, account=account)
    return {"account": account, "count": len(history), "history": history}


@app.delete("/api/account/{account}/history")
def delete_account_history(account: str = "mini"):
    clear_account_history(BASE_DIR, account=account)
    return {"account": account, "count": 0, "history": []}


@app.post("/api/analyze")
async def analyze_uploaded_image(
    file: UploadFile = File(...),
    account: str = Query(default="mini"),
):
    content_type = (file.content_type or "").lower()
    if content_type and not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only still image files (JPG, PNG, WEBP) are accepted. Video inputs are disabled.")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 12 MB maximum forensic buffer.")

    try:
        report = await analyze_image_pipeline(raw_bytes, filename=file.filename or "uploaded_image.png")
        save_to_account_history(BASE_DIR, report, account=account)
        return report
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not decode or analyze image: {exc}") from exc


@app.post("/api/analyze-sample/{sample_id}")
async def analyze_sample(sample_id: str, account: str = Query(default="mini")):
    ensure_sample_images(SAMPLES_DIR)
    safe_name = Path(sample_id).name
    target = SAMPLES_DIR / safe_name
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Sample image not found.")
    raw_bytes = target.read_bytes()
    report = await analyze_image_pipeline(raw_bytes, filename=safe_name)
    save_to_account_history(BASE_DIR, report, account=account)
    return report


app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
