"""FastAPI application factory and REST endpoints for Tree of Life."""
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional
from datetime import datetime, timezone, timedelta
from io import BytesIO
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.schemas.response import PredictResponse, PartialResult
from app.schemas.variety import VarietyResolution
from app.schemas.genomic import GenomicResult
from app.core.config import settings
from app.db.mongo import db_manager
from app.pipeline.orchestrator import orchestrator
from app.pipeline.storage import storage_service
from app.pipeline.janitor import janitor
from app.services.variety.resolver import VarietyResolver
from app.services.genomic.store import GenomicStore
from app.pipeline.gpu_lock import gpu_lock


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Connect MongoDB
    try:
        await db_manager.connect()
        print(f"[Startup] Connected to MongoDB at {settings.MONGO_URI}")
    except Exception as e:
        print(f"[Startup] MongoDB connection warning: {e}. Running in memory-buffered mode.")

    # Run janitor on startup
    try:
        cleaned = await janitor.cleanup_expired_records()
        if cleaned > 0:
            print(f"[Janitor] Cleaned {cleaned} expired records on startup.")
    except Exception:
        pass

    yield

    # Teardown
    await db_manager.close()


app = FastAPI(
    title="Tree of Life - Rice Disease Decision Support",
    description="Multi-Modal Crop-Disease Decision Support Backend for Indian Smallholder Farmers",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

variety_resolver = VarietyResolver()
genomic_store = GenomicStore()


# -------------------------------------------------------------------
# Predict Endpoint
# -------------------------------------------------------------------
@app.post("/api/v1/disease/predict", response_model=PredictResponse, status_code=status.HTTP_200_OK)
async def predict_disease(
    leaf_images: List[UploadFile] = File(...),
    seed_packet: Optional[UploadFile] = File(None),
    variety_text: Optional[str] = Form(None),
    district: Optional[str] = Form(None),
    crop: str = Form("rice"),
    wait: bool = Query(True)
):
    if len(leaf_images) < 3 or len(leaf_images) > 5:
        raise HTTPException(
            status_code=400,
            detail=f"Between 3 and 5 leaf images must be provided (received {len(leaf_images)})."
        )

    # Process and strip EXIF from images
    pil_images: List[Image.Image] = []
    for file in leaf_images:
        content = await file.read()
        dest_path, file_id, expires_at = storage_service.save_and_strip_exif(content, file.filename or "leaf.jpg")
        pil_images.append(Image.open(dest_path))

    seed_pil = None
    if seed_packet:
        sp_content = await seed_packet.read()
        sp_path, _, _ = storage_service.save_and_strip_exif(sp_content, seed_packet.filename or "packet.jpg")
        seed_pil = Image.open(sp_path)

    # Execute orchestrator pipeline
    response = await orchestrator.run_pipeline(
        leaf_images=pil_images,
        seed_packet_image=seed_pil,
        variety_text=variety_text,
        district=district,
        crop=crop
    )

    return response


# -------------------------------------------------------------------
# Result & Status Endpoint
# -------------------------------------------------------------------
@app.get("/api/v1/disease/result/{request_id}", response_model=PredictResponse)
async def get_prediction_result(request_id: str):
    if db_manager.async_db is None:
        raise HTTPException(status_code=404, detail="Database not initialized.")

    doc = await db_manager.async_db.prediction_requests.find_one({"request_id": request_id})
    if not doc:
        raise HTTPException(status_code=404, detail=f"Prediction request {request_id} not found.")

    doc.pop("_id", None)
    return PredictResponse.model_validate(doc)


# -------------------------------------------------------------------
# Variety Resolution Endpoint
# -------------------------------------------------------------------
class VarietyResolveRequest(BaseModel):
    query: str
    source: str = "farmer_dropdown"


@app.post("/api/v1/variety/resolve", response_model=VarietyResolution)
async def resolve_variety(req: VarietyResolveRequest):
    return variety_resolver.resolve(variety_text=req.query)


# -------------------------------------------------------------------
# Genotype Profile Endpoint
# -------------------------------------------------------------------
@app.get("/api/v1/genotype/profile/{canonical_variety_id}", response_model=GenomicResult)
async def get_genotype_profile(canonical_variety_id: str):
    return genomic_store.get_profile(canonical_variety_id)


# -------------------------------------------------------------------
# Expert Escalation Endpoint (Requires explicit user consent)
# -------------------------------------------------------------------
class ExpertEscalateRequest(BaseModel):
    request_id: str
    consent_given: bool
    include_images: bool = False
    notes: Optional[str] = None


@app.post("/api/v1/expert/escalate", status_code=status.HTTP_201_CREATED)
async def escalate_to_expert(req: ExpertEscalateRequest):
    if not req.consent_given:
        raise HTTPException(
            status_code=400,
            detail="Explicit user consent (consent_given=True) is required to escalate data to expert pathologists."
        )

    escalation_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    retention_expires_at = now + timedelta(days=settings.ESCALATION_RETENTION_DAYS)

    doc = {
        "escalation_id": escalation_id,
        "request_id": req.request_id,
        "consent_given": req.consent_given,
        "include_images": req.include_images,
        "notes": req.notes,
        "status": "open",
        "created_at": now.isoformat(),
        "retention_expires_at": retention_expires_at.isoformat()
    }

    if db_manager.async_db is not None:
        await db_manager.async_db.expert_escalations.insert_one(doc)

    return {
        "escalation_id": escalation_id,
        "status": "open",
        "retention_days": settings.ESCALATION_RETENTION_DAYS,
        "message": "Case successfully escalated to agricultural extension review."
    }


# -------------------------------------------------------------------
# Models Health & Diagnostic Endpoint
# -------------------------------------------------------------------
@app.get("/api/v1/models/health")
async def get_models_health():
    vram_used = "0 MB"
    try:
        import torch
        if torch.cuda.is_available():
            vram_used = f"{torch.cuda.memory_allocated() / (1024**2):.1f} MB"
    except Exception:
        pass

    return {
        "device": settings.DEVICE,
        "gpu_lock_owner": gpu_lock.current_owner,
        "vram_allocated": vram_used,
        "database": {
            "type": settings.DATABASE_TYPE,
            "connected": db_manager.async_db is not None,
            "uri": settings.MONGO_URI,
            "name": settings.MONGO_DB_NAME
        },
        "adapters": {
            "florence": "real (Florence-2-large)" if settings.USE_REAL_FLORENCE else "mock/classical",
            "qwen": "real (Qwen2.5-VL-3B-NF4)" if settings.USE_REAL_QWEN else "mock",
            "efficientnet": "real (EfficientNet-B0)" if settings.USE_REAL_EFFICIENTNET else "mock"
        }
    }
