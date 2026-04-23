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

export interface RobustnessStatus {
  passed: boolean;
  detail: string;
  flagged_tokens: string[];
}

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
