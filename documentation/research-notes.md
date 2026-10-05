# TREE OF LIFE — Research Notes (distilled)

Purpose: give Kiro (and humans) the *design-relevant* lessons from the four reference papers without pasting the PDFs.
Keep the PDFs outside the repo or in `docs/references/` (not referenced by steering/specs).

## 1. Garnica & Ojiambo 2025 — window-pane weather analysis (Front. Plant Sci. 16:1637130)
**What it did:** wheat Stagonospora nodorum blotch; 18 environments; hourly T / RH / rain turned into 1,530 engineered time series (hours in temperature bands, hours RH >= 80/90 %, rain-event counts, dawn/dusk/daytime windows, 5–30 day rolling windows relative to anthesis); stability selection (LASSO + bootstrap) to pick stable predictors; weather data gaps filled from ERA5 via the free Open-Meteo API.
**Take:**
- Feature *schema* for the environment branch: counts of hours in conditions (e.g. RH >= 90 %, T in 25–30 C, rainfall hours) over rolling windows aligned to crop growth stage.
- Free weather source: Open-Meteo (no paid API). Cache by district + date.
- Genotype-by-environment interaction (GEI) is real: the same variety behaves differently across environments. A variety-level prior is therefore a *weak, context-dependent* prior.
**Do not take:** the wheat-specific predictors, or the reported correlations (r ~ 0.4–0.7, n = 18, authors themselves warn about multiple testing and autocorrelated windows). Rice disease–weather rules must come from agronomist-validated rules first, learned models only after Tree of Life has its own labelled data.
**Design consequence:** environment weight stays 0.05, rule-based risk curves, never overrides visual evidence.

## 2. Kajrolkar 2025 — "Deep Learning Applications in Plant Biology" (Premier J. Plant Biol.)
**Reliability warning:** treat as background reading only. Several citations do not match the claims they support (e.g. an epidermal-growth-factor gene-family paper is cited for a CNN phenotyping model; an ExPecto claim is cited to a plant-phenomics paper), the text is partly garbled, and the "proposed" architectures (Plant Transformer, Bio Hybrid Net, EcoSystemNet, PlantMorphoNet) have no results.
**Take (general ideas only):** genotype + phenotype + environment integration is a legitimate research direction; interpretability, data quality, and generalisation across environments are the main risks.
**Do not cite:** its accuracy numbers (e.g. ">99 % on PlantVillage"). Lab-style datasets do not transfer to field photos — which is why Tree of Life keeps an untouched field test set.

## 3. Sánchez et al. 2022 — Bio-SCAN CRISPR lateral-flow assay (Plant Biotechnol. J. 20:2418)
**What it did:** wet-lab, field-deployable (< 1 h) detection of specific alleles and pathogen DNA/RNA using RPA amplification + biotinylated dCas9 + sgRNA + lateral-flow strip. Detected wheat Lr34/Lr67 resistance alleles, engineered herbicide-resistance alleles in rice, transgene promoters, and several pathogens.
**Take:**
- This is the *honest* route to real genotype evidence: a laboratory/strip result for a specific allele on a specific sample.
- Reserve a source type in the schema now: `lab_allele_assay` (unimplemented in Phase 1). It would outrank any variety-level lookup.
- Also a possible future route for variety verification and pathogen confirmation of ambiguous cases.
**Limits:** each target needs its own primers + sgRNA; needs reagents (recombinant bio-dCas9, RPA kit); detects named alleles, not polygenic susceptibility; not a farmer-facing tool; none of the tested targets are rice disease-resistance genes. Not a Phase 1 dependency.

## 4. Chadoulis et al. 2025 — 3D-CNN hyperspectral Potexvirus detection (Plant Methods 21:15)
**What it did:** 60 *N. benthamiana* plants, lab hyperspectral camera, binary healthy/diseased, 16x16 patches, plant-level 3-fold CV, accuracy 0.78–0.87 in-genotype, 0.48–0.83 on unseen genotypes.
**Take:**
- **Group-aware splitting:** split by plant (here: by plant/farm/source), never by image. Mandatory in Tree of Life.
- **Patch/crop idea:** many local crops per image increases training signal and gives localisation — same logic as Florence lesion crops → EfficientNet.
- **Genotype shift hurts accuracy** → evaluate leave-one-variety-out and report per-variety metrics.
- **Data-integrity checks** before training (dimension/corruption checks, a logged error CSV) → implement `scripts/validate_dataset.py`.
**Do not take:** hyperspectral hardware (not available to farmers), or the reported accuracies as a benchmark for RGB field photos. Tungro is a virus; early/pre-symptomatic detection is out of scope for Phase 1.

## Cross-cutting lessons
1. Lab accuracy != field accuracy. Report both, never merge them.
2. Group/source-aware splits and per-variety evaluation are non-negotiable.
3. Small n inflates correlations; require ablations before claiming fusion helps.
4. Genomic information is only as good as the phenotype labels used to learn from it; with ~6 target varieties, a learned genomic model is not statistically meaningful — use a curated lookup in Phase 1.
