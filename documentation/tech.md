---
inclusion: always
---

# TREE OF LIFE — Tech Steering

## Stack
Python 3.11, PyTorch + torchvision, Hugging Face Transformers, FastAPI + Uvicorn, Pydantic v2 + pydantic-settings, SQLAlchemy 2.0 + Alembic (SQLite now, PostgreSQL-ready: generic `JSON` type with `JSONB` variant, no SQLite-only SQL), pytest + httpx, ruff, mypy. React dashboard comes later. No paid APIs, no cloud calls at inference time. Optional free weather: Open-Meteo, cached.

## Hardware budget (hard constraints)
RTX 4050 Laptop 6 GB VRAM, i5-12450H, 16 GB RAM, Windows 64-bit, limited disk.
- Never hold Florence-2-large and Qwen2.5-VL-3B on the GPU at the same time.
- Florence-2-large: FP16. Qwen2.5-VL-3B-Instruct: 4-bit NF4 (bitsandbytes), fp16 compute. EfficientNet-B0: FP16 (tiny; may stay resident).
- DNABERT-2 is OFFLINE ONLY. Never imported by the API process.
- After each heavy model: `del model; gc.collect(); torch.cuda.empty_cache()`; log VRAM before/after.
- Cap Qwen image tokens (set `min_pixels`/`max_pixels`; start around 256-512 x 28 x 28 per image), `max_new_tokens` ~512, greedy decoding. Measure; do not assume.
- CPU fallback for every model: detected automatically if CUDA is missing or OOM occurs; mark `device="cpu"` and `degraded=true` in audit.
- Latency numbers (quality < 1 s, Florence < 3 s, Qwen < 5 s, total < 12 s) are ASPIRATIONAL targets. Measure on the real laptop and record in `docs/latency.md`. Model load time counts.
- Cache HF models under `HF_HOME` configured in `.env`; support `HF_HUB_OFFLINE=1`.

## GPU scheduling
- One process owns the GPU. A single `GpuLock` (async-safe) serialises heavy stages; jobs run through an in-process async queue with bounded size (return 429/503 when full).
- Residency policy (config `MODEL_RESIDENCY`): `on_demand` (load/unload per job) or `batch_stage` (drain queued jobs per stage so one model load serves several jobs).
- Windows: use `spawn` semantics, `pathlib`, no `fork`, no shell-specific paths, no `flash_attn` dependency (use `attn_implementation="eager"` or `sdpa`). Check whether the installed `transformers` has native Florence-2 support; otherwise `trust_remote_code=True` with a pinned revision.

## Service pattern (mandatory)
Every model-backed capability is a `Protocol` in `app/services/interfaces.py` with: a real implementation, a deterministic **mock** implementation (CPU, no downloads), and a factory chosen by config flags (`USE_REAL_FLORENCE`, `USE_REAL_QWEN`, `USE_REAL_EFFICIENTNET`). The whole pipeline and the full test suite must run end-to-end with mocks and no GPU.
Heavy imports (`torch`, `transformers`) are lazy, inside real implementations only.

## Structured output
- Qwen output validated by Pydantic. One JSON-repair retry, then `qwen_phenotype_available=false`; never fail the request. Prefer constrained decoding if a lightweight option works on Windows; otherwise validate + repair.
- All API responses are Pydantic models; no raw dicts crossing module boundaries.

## Calibration, OOD, evaluation
- Temperature scaling on a validation set; report ECE and Brier. Never treat raw softmax as truth.
- OOD: energy / max-logit / embedding-distance score + explicit `unknown_or_ood` class; do not rely on temperature scaling for OOD.
- Splits are grouped by source/farm/plant, never by image. Field images form an untouched external test set. Report lab and field metrics separately, plus per-variety metrics.
- Fusion improvement is claimed only if an ablation in `ml/eval` shows it.

## Code standards
Type hints everywhere, `ruff` + `mypy` clean, docstrings on public functions, structured logging (no PII, no image bytes), config only via `pydantic-settings` (`.env`), no secrets in repo, deterministic seeds in tests, small pure functions for fusion/routing so they are trivially unit-testable.

## Dependencies
Split files: `requirements.txt` (runtime), `requirements-dev.txt`, `requirements-research.txt` (xgboost, pysam, DNABERT-2 stack; never imported by the API). Pin compatible ranges, verify torch/torchvision/transformers/bitsandbytes/accelerate compatibility on Windows, and document the exact PyTorch CUDA install command in the README (taken from pytorch.org at build time).
