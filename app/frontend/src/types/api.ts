export interface Claim {
  text: string;
  confidence: number;
  confidence_level: 'high' | 'moderate' | 'low';
  guidance: string;
}

export interface SecurityResample {
  index: number;
  old_token: string;
  new_token: string;
  severity: number;
}

export interface SecurityStatus {
  certified: boolean | null;
  tokens: string[];
  resampled: SecurityResample[];
}

export interface AdversarialWorstCase {
  suffix: string;
  clean_response: string;
  adv_response: string;
  flipped: boolean;
  score: number | null;
}

export type RobustnessStatus =
  | { type: 'unavailable' }
  | {
      type: 'nlp' | 'mcq';
      attempts: number;
      flipped: number;
      worst_case: AdversarialWorstCase | null;
    };

export interface Uncertainty {
  overall: number | null;
}

export interface AnalysisResponse {
  claims: Claim[];
  overall_confidence: number;
  uncertainty: Uncertainty;
  security: SecurityStatus;
  robustness: RobustnessStatus;
  raw_response: string;
  model: string;
  is_mcq: boolean | null;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  analysis?: AnalysisResponse;
}
