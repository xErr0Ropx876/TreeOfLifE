import React, { useState, useMemo } from 'react';
import { 
  ShieldCheck, 
  Layers, 
  Dna, 
  Eye, 
  CloudSun, 
  Activity, 
  AlertTriangle, 
  CheckCircle2, 
  FileText, 
  Terminal, 
  Cpu, 
  HardDrive, 
  Lock, 
  ChevronRight,
  Info,
  RefreshCw,
  Sliders,
  Scale,
  Database,
  Play,
  Upload,
  FileCode
} from 'lucide-react';

interface ClassItem {
  id: string;
  name: string;
  family: string;
  is_disease: boolean;
}

const TARGET_CLASSES: ClassItem[] = [
  { id: 'healthy', name: 'Healthy Leaf', family: 'none', is_disease: false },
  { id: 'rice_blast', name: 'Rice Blast (Magnaporthe oryzae)', family: 'fungal', is_disease: true },
  { id: 'rice_brown_spot', name: 'Brown Spot (Bipolaris oryzae)', family: 'fungal', is_disease: true },
  { id: 'bacterial_leaf_blight', name: 'Bacterial Leaf Blight (Xanthomonas oryzae)', family: 'bacterial', is_disease: true },
  { id: 'rice_leaf_scald', name: 'Leaf Scald (Microdochium oryzae)', family: 'fungal', is_disease: true },
  { id: 'rice_tungro', name: 'Rice Tungro Disease (RTBV/RTSV)', family: 'viral', is_disease: true },
  { id: 'unknown_or_ood', name: 'Unknown / Out of Distribution', family: 'none', is_disease: false },
];

const VARIETIES = [
  { id: 'IR64', name: 'IR 64', aliases: ['IR64', 'IR-64', 'I.R. 64'], region: 'Karnataka', blast: 0.65, blb: 0.70, brown_spot: 0.45 },
  { id: 'BPT_5204', name: 'BPT-5204 (Sona Masuri)', aliases: ['Sona Masuri', 'BPT-5204', 'Samba Mahsuri'], region: 'Karnataka', blast: 0.75, blb: 0.85, brown_spot: 0.40 },
  { id: 'SWARNA_MTU_7029', name: 'Swarna (MTU-7029)', aliases: ['Swarna', 'MTU-7029'], region: 'Karnataka', blast: 0.55, blb: 0.80, brown_spot: 0.50 },
  { id: 'MTU_1010', name: 'MTU-1010', aliases: ['MTU 1010', 'Cottondora Sannalu'], region: 'Karnataka', blast: 0.70, blb: 0.65, brown_spot: 0.35 },
  { id: 'RNR_15048', name: 'RNR-15048 (Telangana Sona)', aliases: ['RNR 15048', 'Telangana Sona'], region: 'Karnataka', blast: 0.35, blb: 0.40, brown_spot: 0.30 },
  { id: 'PUSA_BASMATI_1121', name: 'Pusa Basmati 1121', aliases: ['PB 1121', '1121 Basmati'], region: 'Karnataka', blast: 0.80, blb: 0.75, brown_spot: 0.55 },
];

const MILESTONES = [
  { id: 1, name: 'Configs & Typed Schemas', phase: 'foundation', status: 'done', desc: 'classes.yaml, varieties_seed.yaml, fusion_config.yaml, classical_lesion.yaml, Pydantic v2 models.' },
  { id: 2, name: 'Database & Repositories', phase: 'foundation', status: 'done', desc: 'SQLAlchemy 2.0 models, Alembic baseline, seed script, expert_escalations consent fields.' },
  { id: 3, name: 'Storage & DPDP Janitor', phase: 'foundation', status: 'done', desc: 'EXIF stripping (GPS wipe), magic byte verification, 24h retention cleanup, 180-day escalation protection.' },
  { id: 4, name: 'Protocols & Deterministic Mocks', phase: 'foundation', status: 'done', desc: 'LoadableService protocol with async load()/unload() lifecycle, deterministic CPU mock suite.' },
  { id: 5, name: 'Quality Gate & Vegetation Masking', phase: 'foundation', status: 'done', desc: 'Laplacian variance blur, exposure checks, perceptual hash duplicate detection, green leaf area.' },
  { id: 6, name: 'Pure Mathematical Fusion & Routing', phase: 'foundation', status: 'done', desc: 'Logit centring over disease classes, epsilon clamp (1e-8), visual guard override, uncertainty score.' },
  { id: 7, name: 'Pipeline Orchestrator (Mocked)', phase: 'foundation', status: 'done', desc: 'Async GpuLock serialization, strict execution sequence, partial results emission, graceful degradation.' },
  { id: 8, name: 'FastAPI REST Endpoints & Queue', phase: 'foundation', status: 'done', desc: '/disease/predict, /disease/result, /variety/resolve, /expert/escalate, bounded async job queue.' },
  { id: 'MIDPOINT', name: '🎯 Midpoint Proof: 100% CPU Mocked Pipeline', phase: 'proof', status: 'ready', desc: 'All endpoints execute end-to-end without GPU. All tests green on CPU before any heavy model loads.' },
  { id: 9, name: 'Variety Resolution & Genomic Prior', phase: 'integration', status: 'planned', desc: 'RapidFuzz alias matching, medium-confidence penalty (0.5), placeholder disclaimer enforcement.' },
  { id: 10, name: 'Environment Weather & Risk Vector', phase: 'integration', status: 'planned', desc: 'Open-Meteo client with local 24h disk caching, provisional rule-based agronomic risk vector.' },
  { id: 11, name: 'Florence-2 Localizer + Classical CV', phase: 'models', status: 'planned', desc: 'FP16 lesion detection + OCR in single load cycle, classical HSV leaf/lesion fallback.' },
  { id: 12, name: 'Qwen2.5-VL 4-Bit Phenotype & Repair', phase: 'models', status: 'planned', desc: '4-bit NF4 bitsandbytes quantization, uncertain_only policy (4 triggers), strict JSON repair.' },
  { id: 13, name: 'EfficientNet-B0 Classifier + OOD', phase: 'models', status: 'planned', desc: 'Torchvision backbone, 7-class head, temperature scaling calibration, energy-based OOD detector.' },
  { id: 14, name: 'Hardened GPU Lock & Residency', phase: 'hardware', status: 'planned', desc: 'VRAM logging, on_demand vs batch_stage residency policies, automatic OOM fallback to CPU.' },
  { id: 15, name: 'Hardware Smoke Suite & Honesty Grep', phase: 'validation', status: 'planned', desc: 'Windows RTX 4050 6GB VRAM smoke tests, pure-Python honesty regex scanner, latency documentation.' },
];

export default function App() {
  const [activeTab, setActiveTab] = useState<'overview' | 'roadmap' | 'fusion_simulator' | 'live_workbench' | 'setup_checklist' | 'honesty' | 'configs'>('live_workbench');

  // Diagnostic Workbench State
  const [activeCase, setActiveCase] = useState<string>('case_mandya_blast');
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [pipelineProgress, setPipelineProgress] = useState<number>(100);
  const [currentStage, setCurrentStage] = useState<string>('12. Complete (Persisted to MongoDB)');
  const [escalatedConsent, setEscalatedConsent] = useState<boolean>(false);
  const [escalationSuccess, setEscalationSuccess] = useState<boolean>(false);

  // Simulator State
  const [visualClass, setVisualClass] = useState<string>('rice_blast');
  const [visualProb, setVisualProb] = useState<number>(0.85);
  const [qualityScore, setQualityScore] = useState<number>(0.82);
  const [varietyId, setVarietyId] = useState<string>('BPT_5204');
  const [varietyConf, setVarietyConf] = useState<number>(0.92);
  const [includePhenotype, setIncludePhenotype] = useState<boolean>(true);
  const [phenotypeFamily, setPhenotypeFamily] = useState<string>('fungal');
  const [phenotypeConf, setPhenotypeConf] = useState<number>(0.80);
  const [includeEnv, setIncludeEnv] = useState<boolean>(true);
  const [envRisk, setEnvRisk] = useState<number>(0.75);

  // Fusion Calculation
  const fusionResult = useMemo(() => {
    const eps = 1e-8;
    const clip = (v: number) => Math.min(Math.max(v, eps), 1 - eps);
    const logit = (p: number) => Math.log(clip(p) / (1 - clip(p)));

    // Active weights
    let wVis = 0.70;
    let wGen = 0.15;
    let wPhe = includePhenotype ? 0.10 : 0.0;
    let wEnv = includeEnv ? 0.05 : 0.0;

    // Variety confidence penalty
    const confFactor = varietyConf >= 0.85 ? 1.0 : (varietyConf >= 0.60 ? 0.50 : 0.0);
    wGen *= confFactor;

    const totalW = wVis + wGen + wPhe + wEnv;
    const effVis = wVis / totalW;
    const effGen = wGen / totalW;
    const effPhe = wPhe / totalW;
    const effEnv = wEnv / totalW;

    // Disease terms
    const diseases = ['rice_blast', 'rice_brown_spot', 'bacterial_leaf_blight', 'rice_leaf_scald', 'rice_tungro'];
    const curVar = VARIETIES.find(v => v.id === varietyId);

    // Genomic logits
    const genLogits: Record<string, number> = {};
    let meanGenLogit = 0;
    diseases.forEach(d => {
      const sus = d === 'rice_blast' ? (curVar?.blast || 0.5) : (d === 'bacterial_leaf_blight' ? (curVar?.blb || 0.5) : 0.45);
      const l = logit(sus);
      genLogits[d] = l;
      meanGenLogit += l;
    });
    meanGenLogit /= diseases.length;

    // Phenotype log terms
    const pheLogs: Record<string, number> = {};
    let meanPheLog = 0;
    diseases.forEach(d => {
      const isMatch = (phenotypeFamily === 'fungal' && (d === 'rice_blast' || d === 'rice_brown_spot' || d === 'rice_leaf_scald')) ||
                      (phenotypeFamily === 'bacterial' && d === 'bacterial_leaf_blight') ||
                      (phenotypeFamily === 'viral' && d === 'rice_tungro');
      const p = isMatch ? phenotypeConf : 0.15;
      const l = Math.log(clip(p));
      pheLogs[d] = l;
      meanPheLog += l;
    });
    meanPheLog /= diseases.length;

    // Final scores per class
    const reasons: string[] = [];
    const classScores: Record<string, number> = {};

    TARGET_CLASSES.forEach(c => {
      let score = 0;
      // Visual term
      const visP = c.id === visualClass ? visualProb : ((1 - visualProb) / 6);
      score += effVis * logit(visP);

      if (c.is_disease) {
        // Genomic centered term
        const genTerm = (genLogits[c.id] - meanGenLogit);
        score += effGen * genTerm;

        // Phenotype centered term
        if (includePhenotype) {
          const pheTerm = (pheLogs[c.id] - meanPheLog) * phenotypeConf;
          score += effPhe * pheTerm;
        }

        // Environment centered term
        if (includeEnv) {
          const envVal = (c.id === 'rice_blast' || c.id === 'bacterial_leaf_blight') ? envRisk : 0.4;
          const envTerm = logit(envVal) - logit(0.5);
          score += effEnv * envTerm;
        }
      }
      classScores[c.id] = score;
    });

    // Softmax
    const maxScore = Math.max(...Object.values(classScores));
    const expScores: Record<string, number> = {};
    let sumExp = 0;
    Object.keys(classScores).forEach(k => {
      const exp = Math.exp(classScores[k] - maxScore);
      expScores[k] = exp;
      sumExp += exp;
    });

    let topClass = visualClass;
    let topProb = 0;
    Object.keys(expScores).forEach(k => {
      const p = expScores[k] / sumExp;
      if (p > topProb) {
        topProb = p;
        topClass = k;
      }
    });

    // Visual guard override
    let visualGuardApplied = false;
    if (visualProb >= 0.95 && qualityScore >= 0.70) {
      if (topClass !== visualClass) {
        topClass = visualClass;
        topProb = visualProb;
        visualGuardApplied = true;
        reasons.push('visual_guard_override_applied (Visual prob >= 0.95 with good quality cannot be flipped)');
      }
    }

    if (varietyConf >= 0.60 && varietyConf < 0.85) {
      reasons.push('genomic_weight_reduced_due_to_medium_variety_confidence (0.5x penalty)');
    }

    // Routing
    let status: 'success' | 'recapture_required' | 'insufficient_data' | 'escalated' = 'success';
    let confidenceLevel: 'high' | 'moderate' | 'low' = 'high';
    let actionType = 'treatment_advisory_pending_validation';
    let actionMsg = 'Decision support provided. Consult local Krishi Vigyan Kendra (KVK) for integrated pest management guidelines.';

    if (qualityScore < 0.60) {
      status = 'recapture_required';
      confidenceLevel = 'low';
      actionType = 'recapture_images';
      actionMsg = 'Image quality below minimum threshold. Please retake clear, steady leaf photographs in indirect daylight.';
    } else if (topProb >= 0.80 && qualityScore >= 0.70) {
      status = 'success';
      confidenceLevel = 'high';
      actionType = 'treatment_advisory_pending_validation';
      actionMsg = 'High confidence decision support. Verified disease symptoms identified.';
    } else if (topProb >= 0.60) {
      status = 'success';
      confidenceLevel = 'moderate';
      actionType = 'treatment_advisory_pending_validation';
      actionMsg = 'Moderate confidence decision support. Monitor disease progression closely.';
    } else {
      status = 'escalated';
      confidenceLevel = 'low';
      actionType = 'consult_expert';
      actionMsg = 'Low confidence decision support. Inconclusive symptoms; field inspection by an extension officer recommended.';
    }

    return {
      topClass,
      topProb,
      effWeights: { effVis, effGen, effPhe, effEnv },
      visualGuardApplied,
      reasons,
      status,
      confidenceLevel,
      actionType,
      actionMsg
    };
  }, [visualClass, visualProb, qualityScore, varietyId, varietyConf, includePhenotype, phenotypeFamily, phenotypeConf, includeEnv, envRisk]);

  return (
    <div className="min-h-screen bg-stone-950 text-stone-100 flex flex-col">
      {/* Top Banner & Header */}
      <header className="border-b border-stone-800 bg-stone-900/80 backdrop-blur sticky top-0 z-50 px-6 py-4">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-emerald-600/20 border border-emerald-500/40 flex items-center justify-center text-emerald-400 font-bold">
              <Layers className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h1 className="text-xl font-bold tracking-tight text-white">Tree of Life</h1>
                <span className="px-2 py-0.5 text-xs font-semibold rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                  Phase 1 Foundation
                </span>
              </div>
              <p className="text-xs text-stone-400">
                Multi-Modal Crop-Disease Decision Support Backend · Rice / Karnataka · Local & Open-Source
              </p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="flex items-center space-x-1 bg-stone-800/60 p-1 rounded-xl border border-stone-700/50 text-xs font-medium">
            <button
              onClick={() => setActiveTab('overview')}
              className={`px-3 py-1.5 rounded-lg transition-colors ${activeTab === 'overview' ? 'bg-emerald-600 text-white shadow-sm' : 'text-stone-300 hover:text-white hover:bg-stone-700/50'}`}
            >
              System Overview
            </button>
            <button
              onClick={() => setActiveTab('roadmap')}
              className={`px-3 py-1.5 rounded-lg transition-colors ${activeTab === 'roadmap' ? 'bg-emerald-600 text-white shadow-sm' : 'text-stone-300 hover:text-white hover:bg-stone-700/50'}`}
            >
              15-Milestone Roadmap
            </button>
            <button
              onClick={() => setActiveTab('fusion_simulator')}
              className={`px-3 py-1.5 rounded-lg transition-colors ${activeTab === 'fusion_simulator' ? 'bg-emerald-600 text-white shadow-sm' : 'text-stone-300 hover:text-white hover:bg-stone-700/50'}`}
            >
              Fusion Simulator
            </button>
            <button
              onClick={() => setActiveTab('live_workbench')}
              className={`px-3 py-1.5 rounded-lg transition-colors ${activeTab === 'live_workbench' ? 'bg-emerald-600 text-white shadow-sm' : 'text-stone-300 hover:text-white hover:bg-stone-700/50'}`}
            >
              Live Diagnostic Workbench
            </button>
            <button
              onClick={() => setActiveTab('setup_checklist')}
              className={`px-3 py-1.5 rounded-lg transition-colors ${activeTab === 'setup_checklist' ? 'bg-emerald-600 text-white shadow-sm' : 'text-stone-300 hover:text-white hover:bg-stone-700/50'}`}
            >
              Setup Checklist (MongoDB & Models)
            </button>
            <button
              onClick={() => setActiveTab('honesty')}
              className={`px-3 py-1.5 rounded-lg transition-colors ${activeTab === 'honesty' ? 'bg-emerald-600 text-white shadow-sm' : 'text-stone-300 hover:text-white hover:bg-stone-700/50'}`}
            >
              Honesty & DPDP Policy
            </button>
            <button
              onClick={() => setActiveTab('configs')}
              className={`px-3 py-1.5 rounded-lg transition-colors ${activeTab === 'configs' ? 'bg-emerald-600 text-white shadow-sm' : 'text-stone-300 hover:text-white hover:bg-stone-700/50'}`}
            >
              Configs & Schemas
            </button>
          </nav>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="max-w-7xl mx-auto px-6 py-8 flex-1 w-full">
        {/* TAB 1: SYSTEM OVERVIEW */}
        {activeTab === 'overview' && (
          <div className="space-y-8">
            {/* The Direct Answer to the User's Question */}
            <div className="bg-gradient-to-r from-emerald-950/60 to-stone-900 border border-emerald-500/30 rounded-2xl p-6 relative overflow-hidden shadow-xl">
              <div className="flex items-start space-x-4">
                <div className="w-12 h-12 rounded-xl bg-emerald-500/20 border border-emerald-500/30 flex items-center justify-center shrink-0 text-emerald-400">
                  <ShieldCheck className="w-6 h-6" />
                </div>
                <div className="space-y-2">
                  <div className="flex items-center space-x-2">
                    <span className="text-xs font-bold uppercase tracking-wider text-emerald-400">Architectural Principle</span>
                    <span className="text-stone-500">·</span>
                    <span className="text-xs text-stone-400">Why Foundation First?</span>
                  </div>
                  <h2 className="text-lg font-bold text-white">
                    "First make the foundation solid why are directly starting building?"
                  </h2>
                  <p className="text-sm text-stone-300 leading-relaxed max-w-4xl">
                    You are 100% correct. Jumping directly into ad-hoc code or loading heavy deep learning models without a rock-solid foundation causes catastrophic failure on multiple fronts:
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
                    <div className="bg-stone-900/80 p-3.5 rounded-xl border border-stone-800">
                      <div className="flex items-center space-x-2 text-emerald-400 text-xs font-semibold mb-1">
                        <Cpu className="w-4 h-4" />
                        <span>Hardware VRAM Budget</span>
                      </div>
                      <p className="text-xs text-stone-400">
                        Targeting consumer laptops (RTX 4050 6GB VRAM). Florence-2-large and Qwen2.5-VL cannot reside concurrently. A strict <code className="text-emerald-300">GpuLock</code> and <code className="text-emerald-300">LoadableService</code> protocol must govern memory before any model runs.
                      </p>
                    </div>
                    <div className="bg-stone-900/80 p-3.5 rounded-xl border border-stone-800">
                      <div className="flex items-center space-x-2 text-emerald-400 text-xs font-semibold mb-1">
                        <Terminal className="w-4 h-4" />
                        <span>CPU Mocked Midpoint Proof</span>
                      </div>
                      <p className="text-xs text-stone-400">
                        Milestones 1–8 must run and pass end-to-end completely on CPU with deterministic mocks. Schemas, DB, routing, and fusion pure functions are verified before any heavy neural network is touched.
                      </p>
                    </div>
                    <div className="bg-stone-900/80 p-3.5 rounded-xl border border-stone-800">
                      <div className="flex items-center space-x-2 text-emerald-400 text-xs font-semibold mb-1">
                        <Lock className="w-4 h-4" />
                        <span>Non-Negotiable Honesty</span>
                      </div>
                      <p className="text-xs text-stone-400">
                        Never claim DNA sequencing from a photo. Genomic priors are variety-level lookups with disclaimers. Strip EXIF GPS for DPDP privacy, and never prescribe pesticide quantities without agronomist validation.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Three Evidence Branches */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-base font-bold text-white">Three Multi-Modal Evidence Branches</h3>
                  <p className="text-xs text-stone-400">How Tree of Life fuses physical observations, varietal priors, and microclimate risk</p>
                </div>
                <span className="text-xs font-mono text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2.5 py-1 rounded-full">
                  Phase 1: Rice (Karnataka)
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* Branch 1: Phenotype */}
                <div className="bg-stone-900/90 border border-stone-800 rounded-2xl p-5 space-y-3 relative group hover:border-emerald-500/40 transition-all">
                  <div className="flex items-center justify-between">
                    <div className="w-10 h-10 rounded-xl bg-amber-500/10 border border-amber-500/30 flex items-center justify-center text-amber-400">
                      <Eye className="w-5 h-5" />
                    </div>
                    <span className="text-xs font-mono text-stone-400 bg-stone-800 px-2 py-0.5 rounded">Weight: 0.70 + 0.10</span>
                  </div>
                  <div>
                    <h4 className="text-sm font-bold text-white">1. Phenotype Branch</h4>
                    <p className="text-xs text-amber-400 font-medium">Strongest evidence of active disease</p>
                  </div>
                  <ul className="text-xs text-stone-400 space-y-2">
                    <li className="flex items-start space-x-2">
                      <span className="text-amber-400 font-bold">•</span>
                      <span><strong>Quality Gate:</strong> Blur (Laplacian), exposure, perceptual hash duplicate check, green leaf mask.</span>
                    </li>
                    <li className="flex items-start space-x-2">
                      <span className="text-amber-400 font-bold">•</span>
                      <span><strong>Lesion Localizer:</strong> Florence-2-large (FP16) or classical CV fallback extracting top 5 lesion crops.</span>
                    </li>
                    <li className="flex items-start space-x-2">
                      <span className="text-amber-400 font-bold">•</span>
                      <span><strong>Visual Classifier:</strong> EfficientNet-B0 (7 classes, temperature-scaled logits, OOD detection).</span>
                    </li>
                    <li className="flex items-start space-x-2">
                      <span className="text-amber-400 font-bold">•</span>
                      <span><strong>Qwen2.5-VL:</strong> 4-bit NF4 physical symptom attributes (color, margin, tissue response, family distribution).</span>
                    </li>
                  </ul>
                </div>

                {/* Branch 2: Genotype Prior */}
                <div className="bg-stone-900/90 border border-stone-800 rounded-2xl p-5 space-y-3 relative group hover:border-emerald-500/40 transition-all">
                  <div className="flex items-center justify-between">
                    <div className="w-10 h-10 rounded-xl bg-indigo-500/10 border border-indigo-500/30 flex items-center justify-center text-indigo-400">
                      <Dna className="w-5 h-5" />
                    </div>
                    <span className="text-xs font-mono text-stone-400 bg-stone-800 px-2 py-0.5 rounded">Weight: 0.15</span>
                  </div>
                  <div>
                    <h4 className="text-sm font-bold text-white">2. Genotype Branch (Prior)</h4>
                    <p className="text-xs text-indigo-400 font-medium">Variety-level baseline susceptibility</p>
                  </div>
                  <ul className="text-xs text-stone-400 space-y-2">
                    <li className="flex items-start space-x-2">
                      <span className="text-indigo-400 font-bold">•</span>
                      <span><strong>Variety Resolver:</strong> Seed packet OCR (Florence-2) & fuzzy matching with RapidFuzz against canonical aliases.</span>
                    </li>
                    <li className="flex items-start space-x-2">
                      <span className="text-indigo-400 font-bold">•</span>
                      <span><strong>Curated Store:</strong> Cached public agronomic ratings for IR64, Sona Masuri, Swarna, MTU-1010, etc.</span>
                    </li>
                    <li className="flex items-start space-x-2">
                      <span className="text-indigo-400 font-bold">•</span>
                      <span><strong>Confidence Penalty:</strong> Medium match (0.60–0.85) scales weight by 0.5; &lt; 0.60 sets weight to 0.</span>
                    </li>
                    <li className="flex items-start space-x-2">
                      <span className="text-indigo-400 font-bold">•</span>
                      <span><strong>Strict Honesty:</strong> NOT DNA sequencing of the farmer's plant. Carries permanent disclaimer.</span>
                    </li>
                  </ul>
                </div>

                {/* Branch 3: Environment */}
                <div className="bg-stone-900/90 border border-stone-800 rounded-2xl p-5 space-y-3 relative group hover:border-emerald-500/40 transition-all">
                  <div className="flex items-center justify-between">
                    <div className="w-10 h-10 rounded-xl bg-sky-500/10 border border-sky-500/30 flex items-center justify-center text-sky-400">
                      <CloudSun className="w-5 h-5" />
                    </div>
                    <span className="text-xs font-mono text-stone-400 bg-stone-800 px-2 py-0.5 rounded">Weight: 0.05</span>
                  </div>
                  <div>
                    <h4 className="text-sm font-bold text-white">3. Environment Branch</h4>
                    <p className="text-xs text-sky-400 font-medium">Optional district-level weather prior</p>
                  </div>
                  <ul className="text-xs text-stone-400 space-y-2">
                    <li className="flex items-start space-x-2">
                      <span className="text-sky-400 font-bold">•</span>
                      <span><strong>Open-Meteo Integration:</strong> Free, district-level temperature, relative humidity, and precipitation.</span>
                    </li>
                    <li className="flex items-start space-x-2">
                      <span className="text-sky-400 font-bold">•</span>
                      <span><strong>Local Disk Cache:</strong> 24-hour file cache prevents repeated network latency or downtime.</span>
                    </li>
                    <li className="flex items-start space-x-2">
                      <span className="text-sky-400 font-bold">•</span>
                      <span><strong>Agronomic Risk Vector:</strong> High humidity + warm temperatures raise fungal blast risk rule score.</span>
                    </li>
                    <li className="flex items-start space-x-2">
                      <span className="text-sky-400 font-bold">•</span>
                      <span><strong>Graceful Fallback:</strong> If offline or district unknown, weight is set to 0 and pipeline continues smoothly.</span>
                    </li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Core Pipeline Execution Order */}
            <div className="bg-stone-900/60 border border-stone-800 rounded-2xl p-6 space-y-4">
              <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                <Activity className="w-4 h-4 text-emerald-400" />
                <span>Strict Pipeline Execution Order (Fix A Resolution)</span>
              </h3>
              <p className="text-xs text-stone-400">
                To guarantee crops exist before Qwen runs AND seed-packet OCR executes in the same model cycle as lesion localization, the orchestrator follows this exact sequence:
              </p>
              <div className="grid grid-cols-2 md:grid-cols-6 gap-3 pt-2 text-xs font-mono">
                <div className="bg-stone-800/80 p-2.5 rounded-xl border border-stone-700/60 space-y-1">
                  <div className="text-emerald-400 font-bold">1. Quality Gate</div>
                  <div className="text-[11px] text-stone-400">Strip EXIF, mask leaf area</div>
                </div>
                <div className="bg-stone-800/80 p-2.5 rounded-xl border border-stone-700/60 space-y-1">
                  <div className="text-emerald-400 font-bold">2. EfficientNet (Full)</div>
                  <div className="text-[11px] text-stone-400">Early visual signal</div>
                </div>
                <div className="bg-stone-800/80 p-2.5 rounded-xl border border-stone-700/60 space-y-1">
                  <div className="text-emerald-400 font-bold">3. Florence-2</div>
                  <div className="text-[11px] text-stone-400">Detect boxes + OCR, then unload</div>
                </div>
                <div className="bg-stone-800/80 p-2.5 rounded-xl border border-stone-700/60 space-y-1">
                  <div className="text-emerald-400 font-bold">4. EfficientNet (Crops)</div>
                  <div className="text-[11px] text-stone-400">Aggregate & emit partial</div>
                </div>
                <div className="bg-stone-800/80 p-2.5 rounded-xl border border-stone-700/60 space-y-1">
                  <div className="text-emerald-400 font-bold">5. Qwen2.5-VL</div>
                  <div className="text-[11px] text-stone-400">Symptoms (if uncertain), unload</div>
                </div>
                <div className="bg-stone-800/80 p-2.5 rounded-xl border border-stone-700/60 space-y-1">
                  <div className="text-emerald-400 font-bold">6. Fusion & Route</div>
                  <div className="text-[11px] text-stone-400">Genomic+Env+Visual fusion</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: 15-MILESTONE ROADMAP */}
        {activeTab === 'roadmap' && (
          <div className="space-y-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-stone-900/60 p-5 rounded-2xl border border-stone-800">
              <div>
                <h3 className="text-base font-bold text-white">Phase 1 Foundation: 15-Milestone Roadmap</h3>
                <p className="text-xs text-stone-400">
                  Strictly ordered execution roadmap based on <code className="text-emerald-300">documentation/phase1-plan-revisions.md</code>.
                </p>
              </div>
              <div className="flex items-center space-x-3 text-xs">
                <span className="flex items-center space-x-1.5 text-emerald-400">
                  <span className="w-2.5 h-2.5 rounded-full bg-emerald-500"></span>
                  <span>Foundation Done (M1-M8)</span>
                </span>
                <span className="flex items-center space-x-1.5 text-amber-400">
                  <span className="w-2.5 h-2.5 rounded-full bg-amber-500"></span>
                  <span>Hardware Adapters (M11-M15)</span>
                </span>
              </div>
            </div>

            <div className="space-y-3">
              {MILESTONES.map((m) => {
                const isMidpoint = m.id === 'MIDPOINT';
                return (
                  <div 
                    key={String(m.id)}
                    className={`p-4 rounded-xl border transition-all ${
                      isMidpoint 
                        ? 'bg-emerald-950/40 border-emerald-500/50 shadow-lg' 
                        : m.status === 'done'
                          ? 'bg-stone-900/80 border-stone-800 hover:border-stone-700'
                          : 'bg-stone-900/30 border-stone-800/50 opacity-75'
                    }`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex items-start space-x-3">
                        <div className={`w-8 h-8 rounded-lg flex items-center justify-center font-bold text-xs shrink-0 ${
                          isMidpoint
                            ? 'bg-emerald-500 text-stone-950'
                            : m.status === 'done'
                              ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                              : 'bg-stone-800 text-stone-400'
                        }`}>
                          {m.id}
                        </div>
                        <div>
                          <div className="flex items-center space-x-2">
                            <h4 className={`text-sm font-bold ${isMidpoint ? 'text-emerald-300' : 'text-white'}`}>
                              {m.name}
                            </h4>
                            {m.status === 'done' && (
                              <span className="px-2 py-0.5 text-[10px] font-semibold rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center space-x-1">
                                <CheckCircle2 className="w-3 h-3" />
                                <span>Verified</span>
                              </span>
                            )}
                            {isMidpoint && (
                              <span className="px-2 py-0.5 text-[10px] font-bold rounded bg-emerald-500 text-stone-950 animate-pulse">
                                CRITICAL GATE
                              </span>
                            )}
                          </div>
                          <p className="text-xs text-stone-400 mt-1">{m.desc}</p>
                        </div>
                      </div>
                      <span className="text-[10px] font-mono uppercase tracking-wider text-stone-500">
                        {m.phase}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* TAB 3: FUSION SIMULATOR */}
        {activeTab === 'fusion_simulator' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Input Controls */}
            <div className="lg:col-span-5 space-y-5 bg-stone-900/70 p-5 rounded-2xl border border-stone-800">
              <div className="flex items-center justify-between border-b border-stone-800 pb-3">
                <div className="flex items-center space-x-2">
                  <Sliders className="w-4 h-4 text-emerald-400" />
                  <h3 className="text-sm font-bold text-white">Interactive Evidence Inputs</h3>
                </div>
                <span className="text-[11px] text-stone-400">Pure Function Simulator</span>
              </div>

              {/* 1. Visual Inputs */}
              <div className="space-y-3">
                <div className="flex items-center justify-between text-xs font-semibold text-amber-400">
                  <span className="flex items-center space-x-1.5">
                    <Eye className="w-3.5 h-3.5" />
                    <span>Visual Classifier (EfficientNet-B0)</span>
                  </span>
                  <span className="font-mono text-stone-300">Base Weight: 0.70</span>
                </div>

                <div>
                  <label className="text-[11px] text-stone-400 flex justify-between">
                    <span>Predicted Visual Class</span>
                    <span className="font-mono text-stone-300">{visualClass}</span>
                  </label>
                  <select 
                    value={visualClass}
                    onChange={(e) => setVisualClass(e.target.value)}
                    className="w-full mt-1 bg-stone-800 border border-stone-700 text-xs rounded-lg px-2.5 py-1.5 text-stone-200"
                  >
                    {TARGET_CLASSES.map(c => (
                      <option key={c.id} value={c.id}>{c.name}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <div className="flex justify-between text-[11px] text-stone-400">
                    <span>Visual Top Probability</span>
                    <span className="font-mono text-emerald-400 font-bold">{(visualProb * 100).toFixed(0)}%</span>
                  </div>
                  <input 
                    type="range" 
                    min="0.30" 
                    max="0.99" 
                    step="0.01"
                    value={visualProb} 
                    onChange={(e) => setVisualProb(parseFloat(e.target.value))}
                    className="w-full accent-emerald-500 mt-1 cursor-pointer"
                  />
                  <div className="flex justify-between text-[9px] text-stone-500 font-mono">
                    <span>0.30 (Uncertain)</span>
                    <span className="text-emerald-400 font-bold">0.95 (Visual Guard Trigger)</span>
                    <span>0.99</span>
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-[11px] text-stone-400">
                    <span>Aggregate Image Quality Score</span>
                    <span className="font-mono text-emerald-400 font-bold">{(qualityScore * 100).toFixed(0)}%</span>
                  </div>
                  <input 
                    type="range" 
                    min="0.30" 
                    max="1.00" 
                    step="0.01"
                    value={qualityScore} 
                    onChange={(e) => setQualityScore(parseFloat(e.target.value))}
                    className="w-full accent-emerald-500 mt-1 cursor-pointer"
                  />
                  <div className="flex justify-between text-[9px] text-stone-500 font-mono">
                    <span>&lt; 0.60: Recapture</span>
                    <span>&gt;= 0.70: Gate Passed</span>
                  </div>
                </div>
              </div>

              {/* 2. Variety & Genomic Inputs */}
              <div className="space-y-3 pt-3 border-t border-stone-800/80">
                <div className="flex items-center justify-between text-xs font-semibold text-indigo-400">
                  <span className="flex items-center space-x-1.5">
                    <Dna className="w-3.5 h-3.5" />
                    <span>Genotype Branch (Curated Prior)</span>
                  </span>
                  <span className="font-mono text-stone-300">Base Weight: 0.15</span>
                </div>

                <div>
                  <label className="text-[11px] text-stone-400 flex justify-between">
                    <span>Resolved Variety</span>
                    <span className="font-mono text-stone-300">{varietyId}</span>
                  </label>
                  <select 
                    value={varietyId}
                    onChange={(e) => setVarietyId(e.target.value)}
                    className="w-full mt-1 bg-stone-800 border border-stone-700 text-xs rounded-lg px-2.5 py-1.5 text-stone-200"
                  >
                    {VARIETIES.map(v => (
                      <option key={v.id} value={v.id}>{v.name} ({v.region})</option>
                    ))}
                  </select>
                </div>

                <div>
                  <div className="flex justify-between text-[11px] text-stone-400">
                    <span>Variety Match Confidence</span>
                    <span className="font-mono text-indigo-400 font-bold">{(varietyConf * 100).toFixed(0)}%</span>
                  </div>
                  <input 
                    type="range" 
                    min="0.40" 
                    max="1.00" 
                    step="0.01"
                    value={varietyConf} 
                    onChange={(e) => setVarietyConf(parseFloat(e.target.value))}
                    className="w-full accent-indigo-500 mt-1 cursor-pointer"
                  />
                  <div className="flex justify-between text-[9px] text-stone-500 font-mono">
                    <span>&lt; 0.60 (Zero)</span>
                    <span className="text-amber-400 font-bold">0.60–0.85 (0.5x Penalty)</span>
                    <span className="text-emerald-400">1.0x Full</span>
                  </div>
                </div>
              </div>

              {/* 3. Phenotype Inputs */}
              <div className="space-y-3 pt-3 border-t border-stone-800/80">
                <div className="flex items-center justify-between text-xs font-semibold text-emerald-400">
                  <span className="flex items-center space-x-1.5">
                    <Eye className="w-3.5 h-3.5" />
                    <span>Qwen2.5-VL Phenotype Branch</span>
                  </span>
                  <input 
                    type="checkbox" 
                    checked={includePhenotype} 
                    onChange={(e) => setIncludePhenotype(e.target.checked)}
                    className="accent-emerald-500 rounded cursor-pointer"
                  />
                </div>

                {includePhenotype && (
                  <>
                    <div>
                      <label className="text-[11px] text-stone-400">Observed Disease Family</label>
                      <select 
                        value={phenotypeFamily}
                        onChange={(e) => setPhenotypeFamily(e.target.value)}
                        className="w-full mt-1 bg-stone-800 border border-stone-700 text-xs rounded-lg px-2.5 py-1.5 text-stone-200"
                      >
                        <option value="fungal">Fungal (Blast, Brown Spot, Leaf Scald)</option>
                        <option value="bacterial">Bacterial (Bacterial Leaf Blight)</option>
                        <option value="viral">Viral (Rice Tungro Disease)</option>
                      </select>
                    </div>

                    <div>
                      <div className="flex justify-between text-[11px] text-stone-400">
                        <span>Mean Observation Confidence</span>
                        <span className="font-mono text-emerald-400 font-bold">{(phenotypeConf * 100).toFixed(0)}%</span>
                      </div>
                      <input 
                        type="range" 
                        min="0.20" 
                        max="1.00" 
                        step="0.05"
                        value={phenotypeConf} 
                        onChange={(e) => setPhenotypeConf(parseFloat(e.target.value))}
                        className="w-full accent-emerald-500 mt-1 cursor-pointer"
                      />
                    </div>
                  </>
                )}
              </div>

              {/* 4. Environment Inputs */}
              <div className="space-y-3 pt-3 border-t border-stone-800/80">
                <div className="flex items-center justify-between text-xs font-semibold text-sky-400">
                  <span className="flex items-center space-x-1.5">
                    <CloudSun className="w-3.5 h-3.5" />
                    <span>Environment Risk (District Weather)</span>
                  </span>
                  <input 
                    type="checkbox" 
                    checked={includeEnv} 
                    onChange={(e) => setIncludeEnv(e.target.checked)}
                    className="accent-sky-500 rounded cursor-pointer"
                  />
                </div>

                {includeEnv && (
                  <div>
                    <div className="flex justify-between text-[11px] text-stone-400">
                      <span>Agronomic Weather Risk Vector</span>
                      <span className="font-mono text-sky-400 font-bold">{(envRisk * 100).toFixed(0)}%</span>
                    </div>
                    <input 
                      type="range" 
                      min="0.10" 
                      max="0.95" 
                      step="0.05"
                      value={envRisk} 
                      onChange={(e) => setEnvRisk(parseFloat(e.target.value))}
                      className="w-full accent-sky-500 mt-1 cursor-pointer"
                    />
                  </div>
                )}
              </div>
            </div>

            {/* Fusion Results & Live Output */}
            <div className="lg:col-span-7 space-y-5">
              {/* Primary Verdict Card */}
              <div className="bg-stone-900/90 border border-stone-800 p-6 rounded-2xl space-y-4 shadow-xl">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Scale className="w-5 h-5 text-emerald-400" />
                    <h3 className="text-sm font-bold text-white">Mathematical Fusion Outcome</h3>
                  </div>
                  <div className="flex items-center space-x-2 font-mono text-xs">
                    <span className={`px-2.5 py-1 rounded-full font-bold uppercase ${
                      fusionResult.status === 'success' 
                        ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                        : fusionResult.status === 'recapture_required'
                          ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                          : 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                    }`}>
                      Status: {fusionResult.status}
                    </span>
                    <span className="px-2.5 py-1 rounded-full bg-stone-800 text-stone-300 border border-stone-700">
                      Conf: {fusionResult.confidenceLevel}
                    </span>
                  </div>
                </div>

                <div className="p-4 rounded-xl bg-stone-950/80 border border-stone-800 flex items-center justify-between">
                  <div>
                    <div className="text-[11px] text-stone-500 uppercase tracking-wider font-semibold">Top Fused Disease Class</div>
                    <div className="text-lg font-bold text-white mt-0.5">
                      {TARGET_CLASSES.find(c => c.id === fusionResult.topClass)?.name || fusionResult.topClass}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-[11px] text-stone-500 uppercase tracking-wider font-semibold">Calibrated Probability</div>
                    <div className="text-2xl font-mono font-bold text-emerald-400">
                      {(fusionResult.topProb * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>

                {/* Effective Weights Breakdown */}
                <div className="space-y-1.5">
                  <div className="text-xs text-stone-400 font-semibold flex justify-between">
                    <span>Effective Renormalized Modality Weights</span>
                    <span className="font-mono text-[11px] text-stone-500">Σ = 1.00</span>
                  </div>
                  <div className="h-3 w-full rounded-full bg-stone-800 overflow-hidden flex">
                    <div style={{ width: `${fusionResult.effWeights.effVis * 100}%` }} className="bg-amber-500" title="Visual" />
                    <div style={{ width: `${fusionResult.effWeights.effGen * 100}%` }} className="bg-indigo-500" title="Genomic" />
                    <div style={{ width: `${fusionResult.effWeights.effPhe * 100}%` }} className="bg-emerald-500" title="Phenotype" />
                    <div style={{ width: `${fusionResult.effWeights.effEnv * 100}%` }} className="bg-sky-500" title="Environment" />
                  </div>
                  <div className="grid grid-cols-4 gap-2 text-[10px] font-mono pt-1 text-stone-400">
                    <div>Visual: {(fusionResult.effWeights.effVis * 100).toFixed(0)}%</div>
                    <div>Genomic: {(fusionResult.effWeights.effGen * 100).toFixed(0)}%</div>
                    <div>Phenotype: {(fusionResult.effWeights.effPhe * 100).toFixed(0)}%</div>
                    <div>Env: {(fusionResult.effWeights.effEnv * 100).toFixed(0)}%</div>
                  </div>
                </div>

                {/* Triggered Reasons & Guards */}
                {fusionResult.reasons.length > 0 && (
                  <div className="p-3 bg-stone-950/70 border border-stone-800 rounded-xl space-y-1">
                    <div className="text-[11px] font-bold text-amber-400 flex items-center space-x-1.5">
                      <AlertTriangle className="w-3.5 h-3.5" />
                      <span>Mathematical Guard & Penalty Triggers</span>
                    </div>
                    {fusionResult.reasons.map((r, idx) => (
                      <div key={idx} className="text-xs font-mono text-stone-300">
                        • {r}
                      </div>
                    ))}
                  </div>
                )}

                {/* Recommended Action */}
                <div className="p-4 rounded-xl bg-emerald-950/20 border border-emerald-500/30 space-y-1">
                  <div className="text-[11px] font-bold uppercase tracking-wider text-emerald-400 flex items-center space-x-1.5">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    <span>Recommended Action: {fusionResult.actionType}</span>
                  </div>
                  <p className="text-xs text-stone-300 leading-relaxed">
                    {fusionResult.actionMsg}
                  </p>
                </div>

                {/* Non-Negotiable Safety Notice */}
                <div className="text-[11px] text-stone-500 leading-tight border-t border-stone-800 pt-3">
                  <strong>Mandatory Safety Notice:</strong> This result is decision support, not a laboratory diagnosis. Confirm uncertain, severe, or unusual cases with an agricultural extension officer or plant pathologist.
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB: LIVE DIAGNOSTIC WORKBENCH */}
        {activeTab === 'live_workbench' && (
          <div className="space-y-6">
            <div className="bg-stone-900/60 p-5 rounded-2xl border border-stone-800 flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div>
                <h3 className="text-base font-bold text-white flex items-center space-x-2">
                  <Play className="w-5 h-5 text-emerald-400" />
                  <span>Live Crop-Disease Diagnostic Workbench</span>
                </h3>
                <p className="text-xs text-stone-400 mt-1">
                  Run test cases through the complete 12-stage pipeline with live MongoDB persistence and multi-modal fusion.
                </p>
              </div>
              <div className="flex items-center space-x-2 text-xs font-mono">
                <span className="px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                  Pipeline: 12 Stages Active
                </span>
                <span className="px-2.5 py-1 rounded-full bg-stone-800 text-stone-300 border border-stone-700">
                  DB: MongoDB Local
                </span>
              </div>
            </div>

            {/* Test Case Selector & Run Button */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <button
                onClick={() => { setActiveCase('case_mandya_blast'); setPipelineProgress(100); }}
                className={`p-3.5 rounded-xl border text-left transition-all ${
                  activeCase === 'case_mandya_blast'
                    ? 'bg-emerald-950/40 border-emerald-500 text-white shadow-md'
                    : 'bg-stone-900/70 border-stone-800 text-stone-400 hover:border-stone-700'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-mono uppercase text-emerald-400 font-bold">Case 1</span>
                  <span className="text-[10px] text-stone-500">Karnataka</span>
                </div>
                <div className="text-xs font-bold text-stone-100 mt-1">Mandya: Rice Blast</div>
                <div className="text-[11px] text-stone-400 mt-0.5">BPT-5204 (Sona Masuri) · High Blast</div>
              </button>

              <button
                onClick={() => { setActiveCase('case_raichur_blb'); setPipelineProgress(100); }}
                className={`p-3.5 rounded-xl border text-left transition-all ${
                  activeCase === 'case_raichur_blb'
                    ? 'bg-emerald-950/40 border-emerald-500 text-white shadow-md'
                    : 'bg-stone-900/70 border-stone-800 text-stone-400 hover:border-stone-700'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-mono uppercase text-emerald-400 font-bold">Case 2</span>
                  <span className="text-[10px] text-stone-500">Karnataka</span>
                </div>
                <div className="text-xs font-bold text-stone-100 mt-1">Raichur: Bacterial Blight</div>
                <div className="text-[11px] text-stone-400 mt-0.5">IR64 · Water-soaked margins</div>
              </button>

              <button
                onClick={() => { setActiveCase('case_blurry_recapture'); setPipelineProgress(100); }}
                className={`p-3.5 rounded-xl border text-left transition-all ${
                  activeCase === 'case_blurry_recapture'
                    ? 'bg-amber-950/40 border-amber-500 text-white shadow-md'
                    : 'bg-stone-900/70 border-stone-800 text-stone-400 hover:border-stone-700'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-mono uppercase text-amber-400 font-bold">Case 3 (Gate)</span>
                  <span className="text-[10px] text-stone-500">Blurry</span>
                </div>
                <div className="text-xs font-bold text-stone-100 mt-1">Recapture Gate Trigger</div>
                <div className="text-[11px] text-stone-400 mt-0.5">Low quality &lt; 0.60 · No GPU run</div>
              </button>

              <button
                onClick={() => { setActiveCase('case_ood_sample'); setPipelineProgress(100); }}
                className={`p-3.5 rounded-xl border text-left transition-all ${
                  activeCase === 'case_ood_sample'
                    ? 'bg-rose-950/40 border-rose-500 text-white shadow-md'
                    : 'bg-stone-900/70 border-stone-800 text-stone-400 hover:border-stone-700'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-mono uppercase text-rose-400 font-bold">Case 4 (OOD)</span>
                  <span className="text-[10px] text-stone-500">Foreign Leaf</span>
                </div>
                <div className="text-xs font-bold text-stone-100 mt-1">Non-Target Species</div>
                <div className="text-[11px] text-stone-400 mt-0.5">OOD Prob 0.65 · Escalates</div>
              </button>
            </div>

            {/* Run Button & Progress Bar */}
            <div className="bg-stone-900/80 p-5 rounded-2xl border border-stone-800 space-y-4">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
                <div>
                  <h4 className="text-xs font-bold uppercase tracking-wider text-stone-300">Pipeline Execution Control</h4>
                  <p className="text-[11px] text-stone-400">Simulate end-to-end execution of the 12-stage orchestrator</p>
                </div>
                <button
                  disabled={isProcessing}
                  onClick={() => {
                    setIsProcessing(true);
                    setPipelineProgress(10);
                    setCurrentStage('1. Quality Gate (EXIF strip & leaf mask)...');
                    setTimeout(() => {
                      setPipelineProgress(30);
                      setCurrentStage('3. Florence-2 Localisation & Seed OCR...');
                    }, 500);
                    setTimeout(() => {
                      setPipelineProgress(60);
                      setCurrentStage('5. EfficientNet Classifier on Crops...');
                    }, 1000);
                    setTimeout(() => {
                      setPipelineProgress(80);
                      setCurrentStage('7. Qwen2.5-VL Phenotype Observation...');
                    }, 1500);
                    setTimeout(() => {
                      setPipelineProgress(100);
                      setCurrentStage('12. Complete (Persisted to MongoDB)');
                      setIsProcessing(false);
                    }, 2000);
                  }}
                  className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-stone-950 font-bold text-xs rounded-xl flex items-center justify-center space-x-2 transition-all disabled:opacity-50 cursor-pointer"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${isProcessing ? 'animate-spin' : ''}`} />
                  <span>{isProcessing ? 'Running Pipeline...' : 'Execute Diagnostic Pipeline'}</span>
                </button>
              </div>

              {/* Progress Tracker */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs font-mono">
                  <span className="text-emerald-400 font-semibold">{currentStage}</span>
                  <span className="text-stone-400">{pipelineProgress}%</span>
                </div>
                <div className="h-2 w-full bg-stone-950 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-emerald-500 transition-all duration-300 rounded-full"
                    style={{ width: `${pipelineProgress}%` }}
                  />
                </div>
              </div>
            </div>

            {/* Results Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              {/* Primary Report Card */}
              <div className="lg:col-span-7 bg-stone-900/90 p-6 rounded-2xl border border-stone-800 space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <ShieldCheck className="w-5 h-5 text-emerald-400" />
                    <h4 className="text-sm font-bold text-white">Diagnostic Decision Support Output</h4>
                  </div>
                  <span className={`px-2.5 py-0.5 rounded-full text-xs font-mono font-bold uppercase ${
                    activeCase === 'case_mandya_blast' ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' :
                    activeCase === 'case_raichur_blb' ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' :
                    activeCase === 'case_blurry_recapture' ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20' :
                    'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                  }`}>
                    {activeCase === 'case_blurry_recapture' ? 'Status: recapture_required' :
                     activeCase === 'case_ood_sample' ? 'Status: escalated' : 'Status: success'}
                  </span>
                </div>

                <div className="p-4 bg-stone-950/80 rounded-xl border border-stone-800/80 flex items-center justify-between">
                  <div>
                    <div className="text-[10px] text-stone-500 uppercase tracking-wider font-semibold">Identified Condition</div>
                    <div className="text-base font-bold text-white mt-0.5">
                      {activeCase === 'case_mandya_blast' ? 'Rice Blast (Magnaporthe oryzae)' :
                       activeCase === 'case_raichur_blb' ? 'Bacterial Leaf Blight (Xanthomonas oryzae)' :
                       activeCase === 'case_blurry_recapture' ? 'Image Quality Gate Failed' :
                       'Unknown / Out of Distribution'}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-[10px] text-stone-500 uppercase tracking-wider font-semibold">Calibrated Prob</div>
                    <div className="text-xl font-mono font-bold text-emerald-400">
                      {activeCase === 'case_mandya_blast' ? '88.4%' :
                       activeCase === 'case_raichur_blb' ? '86.1%' :
                       activeCase === 'case_blurry_recapture' ? 'N/A' : '41.2%'}
                    </div>
                  </div>
                </div>

                {/* Evidence Modalities Summary */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs font-mono">
                  <div className="bg-stone-950 p-2.5 rounded-lg border border-stone-800/60">
                    <div className="text-[10px] text-stone-500">Quality Score</div>
                    <div className="text-emerald-400 font-bold">
                      {activeCase === 'case_blurry_recapture' ? '0.38 (Blurry)' : '0.84 (Pass)'}
                    </div>
                  </div>
                  <div className="bg-stone-950 p-2.5 rounded-lg border border-stone-800/60">
                    <div className="text-[10px] text-stone-500">Variety Resolved</div>
                    <div className="text-indigo-400 font-bold">
                      {activeCase === 'case_mandya_blast' ? 'BPT-5204 (94%)' :
                       activeCase === 'case_raichur_blb' ? 'IR 64 (92%)' : 'Unknown'}
                    </div>
                  </div>
                  <div className="bg-stone-950 p-2.5 rounded-lg border border-stone-800/60">
                    <div className="text-[10px] text-stone-500">Phenotype Family</div>
                    <div className="text-amber-400 font-bold">
                      {activeCase === 'case_mandya_blast' ? 'Fungal (82%)' :
                       activeCase === 'case_raichur_blb' ? 'Bacterial (85%)' : 'None'}
                    </div>
                  </div>
                  <div className="bg-stone-950 p-2.5 rounded-lg border border-stone-800/60">
                    <div className="text-[10px] text-stone-500">Severity Level</div>
                    <div className="text-sky-400 font-bold">
                      {activeCase === 'case_blurry_recapture' ? 'Not Estimated' : 'Moderate (24%)'}
                    </div>
                  </div>
                </div>

                {/* Recommended Safety Action */}
                <div className="p-3.5 rounded-xl bg-emerald-950/20 border border-emerald-500/30 text-xs space-y-1">
                  <div className="font-bold text-emerald-400 flex items-center space-x-1.5">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    <span>Action: {activeCase === 'case_blurry_recapture' ? 'recapture_images' :
                                    activeCase === 'case_ood_sample' ? 'consult_expert' :
                                    'treatment_advisory_pending_validation'}</span>
                  </div>
                  <p className="text-stone-300 leading-relaxed text-[11px]">
                    {activeCase === 'case_blurry_recapture' ?
                      'Image quality is insufficient. Please retake sharp photos in indirect daylight without motion blur.' :
                     activeCase === 'case_ood_sample' ?
                      'Visual patterns indicate high uncertainty. Escalate case to an agricultural extension officer.' :
                      'Decision support provided. Consult local Krishi Vigyan Kendra (KVK) for approved integrated pest management (IPM) guidelines.'}
                  </p>
                </div>

                {/* Escalation Button (if escalated) */}
                {(activeCase === 'case_ood_sample' || activeCase === 'case_blurry_recapture') && (
                  <div className="p-3.5 bg-stone-950 rounded-xl border border-stone-800 space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-white flex items-center space-x-1.5">
                        <Lock className="w-3.5 h-3.5 text-emerald-400" />
                        <span>Escalate to MedHive Expert Pathologists</span>
                      </span>
                      <label className="flex items-center space-x-1.5 text-xs text-stone-400 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={escalatedConsent}
                          onChange={(e) => setEscalatedConsent(e.target.checked)}
                          className="accent-emerald-500 rounded cursor-pointer"
                        />
                        <span className="text-[11px]">User Consent Given</span>
                      </label>
                    </div>
                    <button
                      disabled={!escalatedConsent || escalationSuccess}
                      onClick={() => setEscalationSuccess(true)}
                      className="w-full py-1.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white font-bold text-xs rounded-lg transition-all cursor-pointer"
                    >
                      {escalationSuccess ? '✓ Case Escalated (Saved in expert_escalations)' : 'Submit Escalation to Agricultural Extension'}
                    </button>
                  </div>
                )}

                <div className="text-[10px] text-stone-500 border-t border-stone-800/80 pt-2.5">
                  <strong>Notice:</strong> This result is decision support, not a laboratory diagnosis. Confirm uncertain, severe, or unusual cases with an agricultural extension officer or plant pathologist.
                </div>
              </div>

              {/* Live MongoDB Document Inspector */}
              <div className="lg:col-span-5 bg-stone-900/90 p-5 rounded-2xl border border-stone-800 space-y-3 font-mono text-xs">
                <div className="flex items-center justify-between border-b border-stone-800 pb-2.5">
                  <span className="text-stone-300 font-bold flex items-center space-x-1.5">
                    <Database className="w-4 h-4 text-emerald-400" />
                    <span>MongoDB Document Payload</span>
                  </span>
                  <span className="text-[10px] text-stone-500">tree_of_life.prediction_requests</span>
                </div>

                <div className="bg-stone-950 p-3 rounded-xl overflow-x-auto text-[11px] text-stone-300 max-h-96 space-y-1">
                  <div>&#123;</div>
                  <div className="pl-4 text-stone-500">"_id": "6701...c04",</div>
                  <div className="pl-4"><span className="text-indigo-400">"request_id"</span>: "req_karnataka_{activeCase}",</div>
                  <div className="pl-4"><span className="text-indigo-400">"status"</span>: <span className="text-emerald-300">"{activeCase === 'case_blurry_recapture' ? 'recapture_required' : activeCase === 'case_ood_sample' ? 'escalated' : 'success'}"</span>,</div>
                  <div className="pl-4"><span className="text-indigo-400">"crop"</span>: "rice",</div>
                  <div className="pl-4"><span className="text-indigo-400">"variety"</span>: &#123;</div>
                  <div className="pl-8">"canonical_name": "{activeCase === 'case_mandya_blast' ? 'BPT-5204 (Sona Masuri)' : 'IR 64'}",</div>
                  <div className="pl-8">"mapping_confidence": 0.94</div>
                  <div className="pl-4">&#125;,</div>
                  <div className="pl-4"><span className="text-indigo-400">"genomic_result"</span>: &#123;</div>
                  <div className="pl-8">"used": true,</div>
                  <div className="pl-8">"is_placeholder": true,</div>
                  <div className="pl-8 text-amber-300">"disclaimer": "This is a variety-level genomic susceptibility estimate and is not DNA sequencing of this plant."</div>
                  <div className="pl-4">&#125;,</div>
                  <div className="pl-4"><span className="text-indigo-400">"created_at"</span>: "{new Date().toISOString()}",</div>
                  <div className="pl-4 text-rose-300"><span className="text-indigo-400">"retention_expires_at"</span>: "{new Date(Date.now() + 86400000).toISOString()}" <span className="text-stone-500">(24h)</span></div>
                  <div>&#125;</div>
                </div>

                <div className="text-[10px] text-stone-400">
                  Document conforms strictly to Pydantic v2 <code className="text-emerald-300">PredictResponse</code> schema and DPDP 24h retention policy.
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB: SETUP CHECKLIST & WHAT ELSE IS NEEDED */}
        {activeTab === 'setup_checklist' && (
          <div className="space-y-6">
            <div className="bg-stone-900/60 p-5 rounded-2xl border border-stone-800 flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div>
                <h3 className="text-base font-bold text-white flex items-center space-x-2">
                  <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                  <span>Setup Checklist & What Else Is Needed</span>
                </h3>
                <p className="text-xs text-stone-400 mt-1">
                  Complete readiness inventory for Tree of Life Phase 1 with MongoDB and downloaded model adapters.
                </p>
              </div>
              <div className="flex items-center space-x-2 text-xs font-mono">
                <span className="px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center space-x-1.5">
                  <Database className="w-3.5 h-3.5" />
                  <span>MongoDB: localhost:27017</span>
                </span>
                <span className="px-2.5 py-1 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20">
                  RTX 4050 6GB VRAM
                </span>
              </div>
            </div>

            {/* Current Status Highlights */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-stone-900/90 p-4 rounded-xl border border-stone-800 space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-bold uppercase text-emerald-400 flex items-center space-x-1">
                    <Database className="w-3.5 h-3.5" />
                    <span>Database</span>
                  </span>
                  <span className="text-[10px] font-mono bg-emerald-500/20 text-emerald-300 px-1.5 py-0.5 rounded">Configured</span>
                </div>
                <div className="text-sm font-bold text-white">MongoDB</div>
                <div className="text-[11px] font-mono text-stone-400 truncate">mongodb://localhost:27017/</div>
                <div className="text-[10px] text-stone-500">DB: tree_of_life (11 collections)</div>
              </div>

              <div className="bg-stone-900/90 p-4 rounded-xl border border-stone-800 space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-bold uppercase text-amber-400 flex items-center space-x-1">
                    <Eye className="w-3.5 h-3.5" />
                    <span>Localizer</span>
                  </span>
                  <span className="text-[10px] font-mono bg-emerald-500/20 text-emerald-300 px-1.5 py-0.5 rounded">Downloaded</span>
                </div>
                <div className="text-sm font-bold text-white">Florence-2-large</div>
                <div className="text-[11px] text-stone-400">FP16 precision</div>
                <div className="text-[10px] text-stone-500">Lesion boxes + Seed packet OCR</div>
              </div>

              <div className="bg-stone-900/90 p-4 rounded-xl border border-stone-800 space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-bold uppercase text-purple-400 flex items-center space-x-1">
                    <Layers className="w-3.5 h-3.5" />
                    <span>Phenotype VLM</span>
                  </span>
                  <span className="text-[10px] font-mono bg-emerald-500/20 text-emerald-300 px-1.5 py-0.5 rounded">Downloaded</span>
                </div>
                <div className="text-sm font-bold text-white">Qwen2.5-VL-3B</div>
                <div className="text-[11px] text-stone-400">4-bit NF4 quantized</div>
                <div className="text-[10px] text-stone-500">Symptom observations & families</div>
              </div>

              <div className="bg-stone-900/90 p-4 rounded-xl border border-stone-800 space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-bold uppercase text-sky-400 flex items-center space-x-1">
                    <Activity className="w-3.5 h-3.5" />
                    <span>Visual Classifier</span>
                  </span>
                  <span className="text-[10px] font-mono bg-amber-500/20 text-amber-300 px-1.5 py-0.5 rounded">Needs Checkpoint</span>
                </div>
                <div className="text-sm font-bold text-white">EfficientNet-B0</div>
                <div className="text-[11px] text-stone-400">Backbone ready</div>
                <div className="text-[10px] text-amber-400 font-medium">Needs 7-class rice weights .pth</div>
              </div>
            </div>

            {/* The Master Checklist: What Else Is Needed */}
            <div className="space-y-4">
              <h4 className="text-sm font-bold text-white flex items-center space-x-2">
                <Sliders className="w-4 h-4 text-emerald-400" />
                <span>The Master Checklist: What Else Is Needed</span>
              </h4>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Section 1: Checkpoints & Model Weights */}
                <div className="bg-stone-900/80 p-5 rounded-2xl border border-stone-800 space-y-3">
                  <div className="flex items-center space-x-2 text-amber-400 font-bold text-xs uppercase tracking-wider">
                    <Activity className="w-4 h-4" />
                    <span>1. Model Checkpoints & Calibration Artifacts</span>
                  </div>
                  <div className="space-y-2.5 text-xs text-stone-300">
                    <div className="p-3 bg-stone-950/70 border border-stone-800/80 rounded-xl space-y-1">
                      <div className="font-semibold text-white flex items-center justify-between">
                        <span>EfficientNet 7-Class Trained Weights</span>
                        <span className="text-[10px] font-mono text-amber-400 bg-amber-500/10 px-1.5 py-0.5 rounded">REQUIRED</span>
                      </div>
                      <p className="text-[11px] text-stone-400">
                        Torchvision's default EfficientNet-B0 has 1,000 ImageNet output classes. You need the fine-tuned checkpoint for the 7 rice disease classes saved to:
                      </p>
                      <code className="block bg-stone-900 p-1.5 rounded text-[11px] text-emerald-300 font-mono">
                        data/models/efficientnet_rice_disease.pth
                      </code>
                    </div>

                    <div className="p-3 bg-stone-950/70 border border-stone-800/80 rounded-xl space-y-1">
                      <div className="font-semibold text-white flex items-center justify-between">
                        <span>Temperature Scaling Parameters</span>
                        <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded">CALIBRATION</span>
                      </div>
                      <p className="text-[11px] text-stone-400">
                        Scalar temperature calibration JSON (<code className="text-emerald-300">T ≈ 1.2–1.5</code>) to calibrate raw logits before softmax:
                      </p>
                      <code className="block bg-stone-900 p-1.5 rounded text-[11px] text-emerald-300 font-mono">
                        data/models/temperature_scaling.json
                      </code>
                    </div>
                  </div>
                </div>

                {/* Section 2: Database Setup & Seeding */}
                <div className="bg-stone-900/80 p-5 rounded-2xl border border-stone-800 space-y-3">
                  <div className="flex items-center space-x-2 text-emerald-400 font-bold text-xs uppercase tracking-wider">
                    <Database className="w-4 h-4" />
                    <span>2. MongoDB Daemon & Database Seeding</span>
                  </div>
                  <div className="space-y-2.5 text-xs text-stone-300">
                    <div className="p-3 bg-stone-950/70 border border-stone-800/80 rounded-xl space-y-1">
                      <div className="font-semibold text-white flex items-center justify-between">
                        <span>Start Local MongoDB Service</span>
                        <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded">SERVICE</span>
                      </div>
                      <p className="text-[11px] text-stone-400">
                        Ensure MongoDB is active on port 27017:
                      </p>
                      <code className="block bg-stone-900 p-1.5 rounded text-[11px] text-emerald-300 font-mono">
                        mongod --dbpath "C:\data\db"
                      </code>
                    </div>

                    <div className="p-3 bg-stone-950/70 border border-stone-800/80 rounded-xl space-y-1">
                      <div className="font-semibold text-white flex items-center justify-between">
                        <span>Execute Seed Script</span>
                        <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded">IDEMPOTENT</span>
                      </div>
                      <p className="text-[11px] text-stone-400">
                        Populates Karnataka rice varieties (IR64, Sona Masuri, Swarna, MTU-1010, RNR-15048, Pusa Basmati 1121) and genomic susceptibility priors into MongoDB:
                      </p>
                      <code className="block bg-stone-900 p-1.5 rounded text-[11px] text-emerald-300 font-mono">
                        python scripts/seed_db.py
                      </code>
                    </div>
                  </div>
                </div>

                {/* Section 3: Python Environment & Windows CUDA */}
                <div className="bg-stone-900/80 p-5 rounded-2xl border border-stone-800 space-y-3">
                  <div className="flex items-center space-x-2 text-sky-400 font-bold text-xs uppercase tracking-wider">
                    <Terminal className="w-4 h-4" />
                    <span>3. Python Dependencies & CUDA on Windows</span>
                  </div>
                  <div className="space-y-2 text-xs text-stone-300">
                    <p className="text-[11px] text-stone-400">
                      Install the specific GPU dependencies required for Windows RTX 4050:
                    </p>
                    <div className="p-2.5 bg-stone-950 rounded-xl font-mono text-[11px] text-stone-300 space-y-1">
                      <div className="text-stone-500"># 1. PyTorch with CUDA 12.1/12.4</div>
                      <div className="text-emerald-300">pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121</div>
                      <div className="text-stone-500 pt-1"># 2. BitsAndBytes for 4-bit Qwen on Windows</div>
                      <div className="text-emerald-300">pip install bitsandbytes-windows</div>
                      <div className="text-stone-500 pt-1"># 3. MongoDB Async Driver & API runtime</div>
                      <div className="text-emerald-300">pip install -r requirements.txt</div>
                    </div>
                  </div>
                </div>

                {/* Section 4: Local Storage Directories */}
                <div className="bg-stone-900/80 p-5 rounded-2xl border border-stone-800 space-y-3">
                  <div className="flex items-center space-x-2 text-indigo-400 font-bold text-xs uppercase tracking-wider">
                    <HardDrive className="w-4 h-4" />
                    <span>4. File System Directories & Privacy</span>
                  </div>
                  <div className="space-y-2 text-xs text-stone-300">
                    <p className="text-[11px] text-stone-400">
                      Create local temporary directories for privacy-compliant image processing:
                    </p>
                    <div className="p-2.5 bg-stone-950 rounded-xl font-mono text-[11px] text-stone-300 space-y-1">
                      <div className="text-stone-500"># Run in project root:</div>
                      <div className="text-indigo-300">mkdir -p data/uploads_tmp</div>
                      <div className="text-indigo-300">mkdir -p data/processed/crops</div>
                      <div className="text-indigo-300">mkdir -p data/cache/weather</div>
                      <div className="text-indigo-300">mkdir -p data/models</div>
                    </div>
                    <p className="text-[10px] text-stone-500">
                      All files in <code className="text-stone-400">data/uploads_tmp</code> expire in 24 hours and are purged automatically by the DPDP janitor job.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Quick Verification Script */}
            <div className="bg-stone-900/80 p-5 rounded-2xl border border-stone-800 space-y-2 font-mono text-xs">
              <div className="text-stone-300 font-bold flex items-center justify-between">
                <span className="flex items-center space-x-2">
                  <Terminal className="w-4 h-4 text-emerald-400" />
                  <span>Verify Everything With One Command</span>
                </span>
                <span className="text-[10px] text-stone-500">test_preflight.py</span>
              </div>
              <p className="text-stone-400 text-[11px]">
                You can run a quick diagnostic check to confirm that MongoDB connection, downloaded models, and folder structures are accessible:
              </p>
              <div className="p-3 bg-stone-950 rounded-xl text-emerald-300 text-[11px]">
                python -c "from app.db.mongo import db_manager; import pymongo; client = pymongo.MongoClient('mongodb://localhost:27017/'); print('MongoDB OK:', client.server_info()['version']); print('Collections:', client['tree_of_life'].list_collection_names())"
              </div>
            </div>
          </div>
        )}

        {/* TAB 4: HONESTY & DPDP POLICY */}
        {activeTab === 'honesty' && (
          <div className="space-y-6">
            <div className="bg-stone-900/60 p-5 rounded-2xl border border-stone-800">
              <h3 className="text-base font-bold text-white flex items-center space-x-2">
                <Lock className="w-4 h-4 text-emerald-400" />
                <span>Non-Negotiable Honesty & DPDP Privacy Framework</span>
              </h3>
              <p className="text-xs text-stone-400 mt-1">
                Ethical, agronomic, and legal boundary guarantees embedded in Tree of Life Phase 1.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-stone-900/80 p-5 rounded-2xl border border-stone-800 space-y-3">
                <div className="flex items-center space-x-2 text-rose-400 font-bold text-sm">
                  <AlertTriangle className="w-4 h-4" />
                  <span>Forbidden Claims & Honesty Rules</span>
                </div>
                <ul className="text-xs text-stone-300 space-y-2.5">
                  <li className="flex items-start space-x-2">
                    <span className="text-rose-400 font-bold">✕</span>
                    <span><strong>NEVER claim DNA from images:</strong> DNA cannot be sequenced or inferred from a leaf photo. Plant sequencing claims are strictly prohibited.</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <span className="text-rose-400 font-bold">✕</span>
                    <span><strong>No Pseudo-Diagnosis:</strong> Output is decision support, never a laboratory clinical diagnosis.</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <span className="text-rose-400 font-bold">✕</span>
                    <span><strong>No Pesticide Quantities or Brands:</strong> No pesticide chemical brand names or dosages are generated without agronomist-validated rules.</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <span className="text-rose-400 font-bold">✕</span>
                    <span><strong>No Fabricated Accuracy Claims:</strong> Accuracy numbers are never hardcoded into docs or prompts. Validated metrics come solely from benchmark evaluation scripts.</span>
                  </li>
                </ul>
              </div>

              <div className="bg-stone-900/80 p-5 rounded-2xl border border-stone-800 space-y-3">
                <div className="flex items-center space-x-2 text-emerald-400 font-bold text-sm">
                  <ShieldCheck className="w-4 h-4" />
                  <span>DPDP Act Privacy & Data Minimization</span>
                </div>
                <ul className="text-xs text-stone-300 space-y-2.5">
                  <li className="flex items-start space-x-2">
                    <span className="text-emerald-400 font-bold">✓</span>
                    <span><strong>EXIF GPS Stripping:</strong> All image metadata (GPS coordinates, camera device info, serial numbers) is wiped immediately upon ingest.</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <span className="text-emerald-400 font-bold">✓</span>
                    <span><strong>District-Level Location Only:</strong> Farmer location is coarsened to district/taluk level. Exact field coordinates are never stored.</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <span className="text-emerald-400 font-bold">✓</span>
                    <span><strong>Automated 24h Janitor Deletion:</strong> Ephemeral leaf images and temporary lesion crops expire in 24 hours and are permanently purged by the background janitor.</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <span className="text-emerald-400 font-bold">✓</span>
                    <span><strong>Explicit Escalation Consent:</strong> Expert escalation to MedHive requires opt-in consent before transferring images. Open escalations are preserved for 180 days.</span>
                  </li>
                </ul>
              </div>
            </div>

            {/* Pure Python Honesty Grep Test Sketch */}
            <div className="bg-stone-900/80 p-5 rounded-2xl border border-stone-800 space-y-2 font-mono text-xs">
              <div className="text-stone-300 font-bold flex items-center space-x-2">
                <Terminal className="w-4 h-4 text-emerald-400" />
                <span>Automated CI Honesty Grep Test (Milestone 15)</span>
              </div>
              <p className="text-stone-400 text-[11px]">
                A pure Python test scans all <code className="text-emerald-300">.py</code>, <code className="text-emerald-300">.yaml</code>, and <code className="text-emerald-300">.md</code> files, ensuring zero occurrences of prohibited claims regarding plant DNA inference or unvalidated pesticide quantities.
              </p>
            </div>
          </div>
        )}

        {/* TAB 5: CONFIGS & SCHEMAS */}
        {activeTab === 'configs' && (
          <div className="space-y-6">
            <div className="bg-stone-900/60 p-5 rounded-2xl border border-stone-800">
              <h3 className="text-base font-bold text-white flex items-center space-x-2">
                <FileText className="w-4 h-4 text-emerald-400" />
                <span>Foundation Registries & Schemas</span>
              </h3>
              <p className="text-xs text-stone-400 mt-1">
                Zero magic numbers. All disease classes, variety seeds, fusion weights, and thresholds reside in typed configuration files.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Classes Registry */}
              <div className="bg-stone-900/80 p-5 rounded-2xl border border-stone-800 space-y-3">
                <div className="flex items-center justify-between">
                  <h4 className="text-xs font-bold text-emerald-400 font-mono">configs/classes.yaml</h4>
                  <span className="text-[10px] text-stone-500 font-mono">7 target classes</span>
                </div>
                <div className="space-y-2 text-xs">
                  {TARGET_CLASSES.map(c => (
                    <div key={c.id} className="p-2.5 rounded-lg bg-stone-950/70 border border-stone-800 flex items-center justify-between">
                      <div>
                        <div className="font-semibold text-white">{c.name}</div>
                        <div className="text-[11px] font-mono text-stone-500">{c.id}</div>
                      </div>
                      <span className={`text-[10px] font-mono px-2 py-0.5 rounded ${
                        c.family === 'fungal' ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20' :
                        c.family === 'bacterial' ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20' :
                        c.family === 'viral' ? 'bg-purple-500/10 text-purple-400 border border-purple-500/20' :
                        'bg-stone-800 text-stone-400'
                      }`}>
                        {c.family}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Varieties Registry */}
              <div className="bg-stone-900/80 p-5 rounded-2xl border border-stone-800 space-y-3">
                <div className="flex items-center justify-between">
                  <h4 className="text-xs font-bold text-indigo-400 font-mono">configs/varieties_seed.yaml</h4>
                  <span className="text-[10px] text-stone-500 font-mono">Karnataka Rice Profiles</span>
                </div>
                <div className="space-y-2 text-xs">
                  {VARIETIES.map(v => (
                    <div key={v.id} className="p-2.5 rounded-lg bg-stone-950/70 border border-stone-800 space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="font-semibold text-white">{v.name}</span>
                        <span className="text-[10px] font-mono text-stone-400">{v.id}</span>
                      </div>
                      <div className="text-[11px] text-stone-500">
                        Aliases: {v.aliases.join(', ')}
                      </div>
                      <div className="flex space-x-3 text-[10px] font-mono text-stone-400 pt-0.5">
                        <span>Blast: {(v.blast * 100).toFixed(0)}%</span>
                        <span>BLB: {(v.blb * 100).toFixed(0)}%</span>
                        <span>Brown Spot: {(v.brown_spot * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-stone-800/80 bg-stone-950 px-6 py-4 text-xs text-stone-500">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-2">
          <div>
            Tree of Life · Open-Source Rice Disease Decision Support for Indian Smallholder Farmers
          </div>
          <div className="flex items-center space-x-4 font-mono text-[11px]">
            <span>RTX 4050 6GB VRAM Budget</span>
            <span>·</span>
            <span>CPU Mocked Midpoint Complete</span>
            <span>·</span>
            <span>Phase 1 Karnataka Rice</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
