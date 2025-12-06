import { PredictionInput } from '@/services/api';

export interface ValidationWarning {
  field: keyof PredictionInput;
  message: string;
  severity: 'low' | 'medium' | 'high';
}

export const validatePatientData = (data: PredictionInput): ValidationWarning[] => {
  const warnings: ValidationWarning[] = [];

  // MMSE Score validation
  if (data.MMSE < 20) {
    warnings.push({
      field: 'MMSE',
      message: 'MMSE score is very low (< 20). This is highly concerning and suggests significant cognitive impairment.',
      severity: 'high',
    });
  } else if (data.MMSE < 24) {
    warnings.push({
      field: 'MMSE',
      message: 'MMSE score is below normal range (24-30). This may indicate cognitive decline.',
      severity: 'medium',
    });
  } else if (data.MMSE > 30) {
    warnings.push({
      field: 'MMSE',
      message: 'MMSE score exceeds maximum (30). Please verify this value.',
      severity: 'low',
    });
  }

  // Age validation
  if (data.Age > 100) {
    warnings.push({
      field: 'Age',
      message: 'Age exceeds 100 years. Please verify this value.',
      severity: 'low',
    });
  } else if (data.Age < 18) {
    warnings.push({
      field: 'Age',
      message: 'Age is below 18. This model is designed for adults.',
      severity: 'medium',
    });
  }

  // Functional Assessment
  if (data.FunctionalAssessment < 3) {
    warnings.push({
      field: 'FunctionalAssessment',
      message: 'Functional Assessment score is very low. This indicates significant functional impairment.',
      severity: 'high',
    });
  } else if (data.FunctionalAssessment > 10) {
    warnings.push({
      field: 'FunctionalAssessment',
      message: 'Functional Assessment exceeds maximum (10). Please verify.',
      severity: 'low',
    });
  }

  // ADL Score
  if (data.ADL < 3) {
    warnings.push({
      field: 'ADL',
      message: 'ADL score is very low. This indicates significant difficulty with daily activities.',
      severity: 'high',
    });
  }

  // Blood Pressure
  if (data.SystolicBP > 180) {
    warnings.push({
      field: 'SystolicBP',
      message: 'Systolic BP is very high (> 180). This may indicate hypertension crisis.',
      severity: 'high',
    });
  } else if (data.SystolicBP < 80) {
    warnings.push({
      field: 'SystolicBP',
      message: 'Systolic BP is very low (< 80). Please verify this value.',
      severity: 'medium',
    });
  }

  if (data.DiastolicBP > 120) {
    warnings.push({
      field: 'DiastolicBP',
      message: 'Diastolic BP is very high (> 120). This may indicate severe hypertension.',
      severity: 'high',
    });
  } else if (data.DiastolicBP < 40) {
    warnings.push({
      field: 'DiastolicBP',
      message: 'Diastolic BP is very low (< 40). Please verify this value.',
      severity: 'medium',
    });
  }

  // BMI
  if (data.BMI > 40) {
    warnings.push({
      field: 'BMI',
      message: 'BMI indicates severe obesity (> 40). This is a significant health risk factor.',
      severity: 'medium',
    });
  } else if (data.BMI < 15) {
    warnings.push({
      field: 'BMI',
      message: 'BMI is extremely low (< 15). Please verify this value.',
      severity: 'high',
    });
  }

  // Multiple symptoms check
  const symptomCount = [
    data.MemoryComplaints,
    data.Confusion,
    data.Disorientation,
    data.Forgetfulness,
    data.DifficultyCompletingTasks,
    data.PersonalityChanges,
    data.BehavioralProblems,
  ].filter(v => v === 1).length;

  if (symptomCount >= 5) {
    warnings.push({
      field: 'MemoryComplaints',
      message: `Multiple cognitive symptoms present (${symptomCount}). This pattern is highly concerning.`,
      severity: 'high',
    });
  }

  return warnings;
};

