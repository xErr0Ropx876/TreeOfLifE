---
inclusion: always
---

# TREE OF LIFE — Repository Structure Steering

```
tree-of-life/
├─ .kiro/
│  ├─ steering/            product.md  tech.md  structure.md
│  └─ specs/               one folder per feature spec
├─ docs/                   research-notes.md  architecture.md  claims-policy.md  latency.md
├─ app/
│  ├─ main.py              FastAPI app factory, lifespan, routers
│  ├─ core/                config.py  logging.py  errors.py  gpu.py  security.py  rate_limit.py
│  ├─ api/v1/              predict.py  jobs.py  variety.py  genotype.py  expert.py  health.py
│  ├─ schemas/             common.py quality.py localizer.py phenotype.py visual.py
│  │                       variety.py genomic.py environment.py fusion.py response.py
│  ├─ db/                  base.py models.py session.py repositories.py seed.py  (+ /alembic)
│  ├─ services/
│  │  ├─ interfaces.py     Protocols for every capability
│  │  ├─ quality/          checks.py (resolution, blur, exposure, duplicates, leaf coverage, EXIF strip)
│  │  ├─ localizer/        base.py florence.py classical.py mock.py   (+ OCR in florence.py)
│  │  ├─ phenotype/        base.py qwen.py mock.py  json_repair.py  aggregate.py
│  │  ├─ visual/           base.py efficientnet.py mock.py calibration.py ood.py aggregate.py
│  │  ├─ variety/          resolver.py fuzzy.py barcode.py
│  │  ├─ genomic/          store.py lookup.py
│  │  ├─ environment/      weather.py risk_rules.py
│  │  ├─ fusion/           fuse.py uncertainty.py routing.py reasons.py
│  │  └─ registry/         model_registry.py
│  ├─ pipeline/            orchestrator.py queue.py gpu_lock.py storage.py janitor.py
│  └─ workers/
├─ ml/                     training/ (efficientnet) · eval/ (metrics, ablations) · genomics_offline/ (DNABERT-2, XGBoost)
├─ configs/                classes.yaml thresholds.yaml fusion.yaml varieties_seed.yaml
├─ data/                   (gitignored) raw/ processed/ genotype_cache/ uploads_tmp/ models/
├─ scripts/                download_models.py  validate_dataset.py  seed_db.py  benchmark_latency.py
├─ tests/                  unit/ integration/ fixtures/
├─ .env.example  requirements.txt  requirements-dev.txt  requirements-research.txt  pyproject.toml  README.md
```

## Conventions
- Dependency direction: `api -> pipeline -> services -> schemas`. Services never import `api`. Only `db/repositories.py` touches SQLAlchemy sessions.
- Thresholds, weights, class lists, variety seeds live in `configs/*.yaml`, loaded once into typed settings objects — never magic numbers in code.
- File and module names are snake_case; Pydantic models PascalCase; class IDs are lowercase snake_case strings (`rice_blast`).
- One spec per feature under `.kiro/specs/<feature>/` (requirements.md, design.md, tasks.md). Tasks are small, ordered, and each ends with a passing test.
- Every service has a mock; every mock has a unit test; every real adapter has a smoke test marked `@pytest.mark.gpu` (skipped when CUDA/models are absent).
- Alembic migrations for every schema change; seed data via `scripts/seed_db.py`, idempotent.
