export interface Claim {
  text: string;
  confidence: number;
  confidence_level: 'high' | 'moderate' | 'low';
  guidance: string;
}

export interface SecurityStatus {
  certified: boolean;
  tpa_budget: number | null;
  detail: string;
}

export type RobustnessStatus =
  | { type: 'unavailable' }
  | {
      type: 'nlp';
      attempts: number;
      flipped: number;
      attacked_response: string;
      attack_suffix: string;
    }
  | {
      type: 'mcq';
      attempts: number;
      flipped: number;
      original_choice: string;
      attacked_choice: string;
      attacked_response: string;
      attack_suffix: string;
    };

export interface AnalysisResponse {
  claims: Claim[];
  overall_confidence: number;
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
