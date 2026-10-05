// Common data contracts for Tree of Life Phase 1 Foundation

export type DiseaseClass =
  | 'healthy'
  | 'rice_blast'
  | 'rice_brown_spot'
  | 'bacterial_leaf_blight'
  | 'rice_leaf_scald'
  | 'rice_tungro'
  | 'unknown_or_ood';

export type DiseaseFamily =
  | 'fungal'
  | 'bacterial'
  | 'viral'
  | 'nutrient_deficiency'
  | 'pest_damage'
  | 'unknown'
  | 'none';

export type PredictionStatus =
  | 'success'
  | 'recapture_required'
  | 'insufficient_data'
  | 'escalated';

export type ConfidenceLevel = 'high' | 'moderate' | 'low';

export type ActionType =
  | 'monitor'
  | 'recapture_images'
  | 'consult_expert'
  | 'treatment_advisory_pending_validation';

export type SeverityLevel = 'low' | 'moderate' | 'high' | 'not_estimated';

export interface QualityIssue {
  type: 'resolution_too_low' | 'blurry' | 'underexposed' | 'overexposed' | 'duplicate' | 'insufficient_leaf_coverage';
  message: string;
  severity: 'warning' | 'error';
  imageIndex: number;
}

export interface ImageQualityDetail {
  imageIndex: number;
  width: number;
  height: number;
  blurVariance: number;
  isBlurry: boolean;
  meanBrightness: number;
  isUnderexposed: boolean;
  isOverexposed: boolean;
  leafCoverage: number;
  isInsufficientCoverage: boolean;
  perceptualHash: string;
  isDuplicate: boolean;
  qualityScore: number;
}

export interface QualityResult {
  aggregateScore: number; // 0.0 to 1.0
  imageQualityValid: boolean;
  issues: QualityIssue[];
  imageDetails: ImageQualityDetail[];
  leafMaskAreas: number[]; // pixels
  privacyActions: string[]; // e.g. ["EXIF metadata stripped", "GPS coordinates removed", "location reduced to district"]
  recaptureGuidance: string | null;
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
}

export interface LesionDetection {
  regionId: string;
  imageIndex: number;
  bbox: BoundingBox;
  cropUrl?: string;
  estimatedAffectedPercent?: number;
}

export interface OCRResult {
  detectedText: string;
  matchedVariety?: string;
  confidence: number;
}

export interface VisualPredictionClass {
  disease: DiseaseClass;
  displayName: string;
  probability: number;
  rawLogit: number;
}

export interface VisualResult {
  top3: VisualPredictionClass[];
  allProbabilities: Record<DiseaseClass, number>;
  qualityScore: number;
  lesionCount: number;
  unknownOodProbability: number;
  entropy: number;
  disagreementScore: number; // Jensen-Shannon divergence between full image & crops [0, 1]
  detections: LesionDetection[];
  cropPredictions?: Array<{ regionId: string; topDisease: DiseaseClass; probability: number }>;
}

export interface LesionObservation {
  regionId: string;
  bbox: [number, number, number, number]; // [ymin, xmin, ymax, xmax] normalized
  lesionColor: string;
  lesionShape: string;
  lesionMargin: string;
  surfaceTexture: string;
  tissueResponse: 'chlorotic' | 'necrotic' | 'water_soaked' | 'powdery' | 'wilted' | 'unknown';
  estimatedRegionAffectedPercent: number;
  distribution: 'isolated' | 'scattered' | 'clustered' | 'streaked' | 'diffuse' | 'unknown';
  possibleCauseFamily: DiseaseFamily;
  visibleSymptoms: string[];
  observationConfidence: number;
}

export interface PhenotypeObservations {
  cropGuess: string;
  visualQuality: {
    isUsable: boolean;
    qualityScore: number;
    issues: string[];
  };
  lesionObservations: LesionObservation[];
  wholeLeafObservations: {
    overallAffectedPercent: number;
    dominantSymptomPattern: string;
    needsMoreImages: boolean;
    requestedImages: string[];
  };
  familyDistribution: Record<string, number>;
  meanObservationConfidence: number;
  qwenPhenotypeAvailable: boolean;
}

export interface VarietyResolution {
  canonicalName: string | null;
  displayName: string;
  source: 'seed_packet_ocr' | 'barcode' | 'farmer_dropdown' | 'expert' | 'unknown';
  mappingConfidence: number;
  requiresUserConfirmation: boolean;
  cropVarietyInconsistency: boolean;
  matchedAlias?: string;
}

export interface GenomicResult {
  used: boolean;
  confidence: number;
  sourceType: 'variety_level_public_profile' | 'lab_allele_assay';
  isPlaceholder: boolean;
  sourceDatabase: string;
  sourceVersion: string;
  susceptibilityVector: Record<string, number>; // 5 core diseases: 0.0 resistant to 1.0 susceptible
  disclaimer: string;
}

export interface EnvironmentResult {
  used: boolean;
  weatherAvailable: boolean;
  district: string;
  taluk?: string;
  temperatureC?: number;
  relativeHumidityPct?: number;
  rainfallMm?: number;
  riskVector: Record<string, number>; // 5 core diseases: 0.0 low to 1.0 high
  rulesValidated: boolean; // false in Phase 1
  notice: string;
}

export interface FusionResult {
  disease: DiseaseClass;
  displayName: string;
  severity: SeverityLevel;
  confidenceLevel: ConfidenceLevel;
  calibratedProbability: number;
  uncertaintyScore: number; // 0.0 to 1.0
  modelAgreement: number; // 0.0 to 1.0
  branchWeights: {
    visual: number;
    genomic: number;
    phenotype: number;
    environment: number;
  };
  fusedProbabilities: Record<DiseaseClass, number>;
  reasons: string[];
  predictionSet: null; // Reserved for conformal prediction
}

export interface RecommendedAction {
  actionType: ActionType;
  title: string;
  message: string;
  agronomicGuidelines: string[];
  safetyNotice: string;
}

export interface AuditData {
  modelVersions: Record<string, string>;
  dataSourceVersions: Record<string, string>;
  timestamp: string;
  privacyActions: string[];
  degradedModes: string[];
}

export interface PredictResponse {
  requestId: string;
  status: PredictionStatus;
  crop: string;
  variety: VarietyResolution;
  visualResult: VisualResult;
  phenotypeObservations?: PhenotypeObservations;
  genomicResult: GenomicResult;
  environmentResult: EnvironmentResult;
  fusionResult: FusionResult;
  recommendedAction: RecommendedAction;
  safetyNotice: string;
  audit: AuditData;
  predictionSet: null;
}

export interface ExpertEscalationRequest {
  requestId: string;
  consentGiven: boolean;
  imagesIncluded: boolean;
  farmerNotes: string;
  contactInfo?: string;
  urgency: 'routine' | 'urgent';
  caseSummary: {
    crop: string;
    variety: string;
    suspectedDisease: string;
    severity: string;
    district: string;
    confidenceLevel: string;
  };
}

export interface ExpertEscalationResponse {
  caseId: string;
  requestId: string;
  status: 'open' | 'under_review' | 'resolved' | 'archived';
  submittedAt: string;
  retentionExpiresAt: string;
  message: string;
}
