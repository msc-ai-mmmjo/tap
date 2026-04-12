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

export interface RobustnessStatus {
  passed: boolean;
  detail: string;
  flagged_tokens: string[];
}

export interface AnalysisResponse {
  claims: Claim[];
  overall_confidence: number;
  security: SecurityStatus;
  robustness: RobustnessStatus;
  raw_response: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  analysis?: AnalysisResponse;
}
