import { PredictionInput, PredictionOutput } from '@/services/api';

export interface PredictionHistoryItem {
  id: string;
  timestamp: string;
  patientData: PredictionInput;
  prediction: PredictionOutput;
}

const HISTORY_KEY = 'alzheimer_predictions_history';
const MAX_HISTORY_ITEMS = 50;

export const saveToHistory = (
  patientData: PredictionInput,
  prediction: PredictionOutput
): void => {
  try {
    const history = getHistory();
    const newItem: PredictionHistoryItem = {
      id: Date.now().toString(),
      timestamp: new Date().toISOString(),
      patientData,
      prediction,
    };
    
    history.unshift(newItem);
    
    // Keep only the last MAX_HISTORY_ITEMS
    if (history.length > MAX_HISTORY_ITEMS) {
      history.splice(MAX_HISTORY_ITEMS);
    }
    
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  } catch (error) {
    console.error('Failed to save to history:', error);
  }
};

export const getHistory = (): PredictionHistoryItem[] => {
  try {
    const stored = localStorage.getItem(HISTORY_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch (error) {
    console.error('Failed to load history:', error);
    return [];
  }
};

export const clearHistory = (): void => {
  try {
    localStorage.removeItem(HISTORY_KEY);
  } catch (error) {
    console.error('Failed to clear history:', error);
  }
};

export const deleteHistoryItem = (id: string): void => {
  try {
    const history = getHistory();
    const filtered = history.filter(item => item.id !== id);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(filtered));
  } catch (error) {
    console.error('Failed to delete history item:', error);
  }
};

