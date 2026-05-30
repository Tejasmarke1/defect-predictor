import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000',
});

// Types
export interface AnalyzeRequest {
  repo_url: string;
  since_days: number;
  top_k: number;
  use_hybrid: boolean;
}

export interface SHAPFeature {
  feature_name: string;
  shap_value: number;
  feature_value: number;
  direction: string;
}

export interface FileRiskScore {
  file_path: string;
  risk_score: number;
  risk_label: string;
  rank: number;
  top_shap_features: SHAPFeature[];
  lines_of_code?: number;
  cyclomatic_complexity?: number;
  last_modified_days_ago?: number;
}

export interface AnalyzeResponse {
  job_id: string;
  status: string;
  repo_url: string;
  repo_name: string;
  total_files_analyzed: number;
  buggy_files_predicted: number;
  top_k_results: FileRiskScore[];
  model_used: string;
  warnings?: string[];
}

export interface ExplainResponse {
  file_path: string;
  risk_score: number;
  risk_label: string;
  shap_waterfall: SHAPFeature[];
  plain_english_summary: string;
  similar_files: string[];
  embedding_neighbors: string[];
}

export const analyzeRepo = async (req: AnalyzeRequest): Promise<AnalyzeResponse> => {
  const { data } = await api.post<AnalyzeResponse>('/analyze', req);
  return data;
};

export const explainFile = async (jobId: string, filePath: string): Promise<ExplainResponse> => {
  const { data } = await api.get<ExplainResponse>(`/explain/${jobId}/${encodeURIComponent(filePath)}`);
  return data;
};

export const getExperiments = async () => {
  const { data } = await api.get('/experiments');
  return data;
};

export const getHealth = async () => {
  const { data } = await api.get('/health');
  return data;
};
