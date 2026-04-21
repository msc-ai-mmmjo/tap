export interface Claim {
  text: string;
  confidence: number;
  confidence_level: 'high' | 'moderate' | 'low';
  guidance: string;
}

export interface Uncertainty {
  overall: number;
}

export interface SecurityResample {
  index: number;
  old_token: string;
  new_token: string;
  severity: number;
}

export interface Security {
  tokens: string[];
  resampled: SecurityResample[];
}

export type RobustnessStatus =
  | {
      type: 'nlp';
      bidirectional_score: 0 | 0.5 | 1 | 1.5 | 2;
      attacked_response: string;
    }
  | {
      type: 'mcq';
      flipped: boolean;
      original_choice: string;
      attacked_choice: string;
      attacked_response: string;
    };

export interface AnalysisResponse {
  raw_response: string;
  model: string;
  is_mcq: boolean | null;
  uncertainty: Uncertainty;
  security: Security;
  robustness: RobustnessStatus;
  claims: Claim[];
  overall_confidence: number;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  analysis?: AnalysisResponse;
}
