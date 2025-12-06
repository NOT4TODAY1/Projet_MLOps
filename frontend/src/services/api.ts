const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface PredictionInput {
  Age: number;
  Gender: number;
  Ethnicity: number;
  EducationLevel: number;
  BMI: number;
  Smoking: number;
  AlcoholConsumption: number;
  PhysicalActivity: number;
  DietQuality: number;
  SleepQuality: number;
  FamilyHistoryAlzheimers: number;
  CardiovascularDisease: number;
  Diabetes: number;
  Depression: number;
  HeadInjury: number;
  Hypertension: number;
  SystolicBP: number;
  DiastolicBP: number;
  CholesterolTotal: number;
  CholesterolLDL: number;
  CholesterolHDL: number;
  CholesterolTriglycerides: number;
  MMSE: number;
  FunctionalAssessment: number;
  MemoryComplaints: number;
  BehavioralProblems: number;
  ADL: number;
  Confusion: number;
  Disorientation: number;
  PersonalityChanges: number;
  DifficultyCompletingTasks: number;
  Forgetfulness: number;
}

export interface PredictionOutput {
  prediction: number;
  diagnosis: string;
  probabilities?: {
    healthy: number;
    alzheimer: number;
  };
}

export interface HealthStatus {
  status: string;
  model_loaded: boolean;
  model_path: string;
  model_type: string;
  scaler_loaded: boolean;
}

export interface RetrainOutput {
  status: string;
  message: string;
}

export const api = {
  async predict(data: PredictionInput): Promise<PredictionOutput> {
    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: `HTTP error! status: ${response.status}` }));
        throw new Error(error.detail || `HTTP error! status: ${response.status}`);
      }

      return response.json();
    } catch (err) {
      if (err instanceof TypeError && err.message.includes('fetch')) {
        throw new Error('Network error: Unable to connect to the API. Please ensure the backend server is running on http://localhost:8000');
      }
      throw err;
    }
  },

  async getHealth(): Promise<HealthStatus> {
    const response = await fetch(`${API_BASE_URL}/health`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
  },

  async retrain(): Promise<RetrainOutput> {
    try {
      const response = await fetch(`${API_BASE_URL}/retrain`, {
        method: 'POST',
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: `HTTP error! status: ${response.status}` }));
        throw new Error(error.detail || `HTTP error! status: ${response.status}`);
      }

      return response.json();
    } catch (err) {
      if (err instanceof TypeError && err.message.includes('fetch')) {
        throw new Error('Network error: Unable to connect to the API. Please ensure the backend server is running on http://localhost:8000');
      }
      throw err;
    }
  },
};

