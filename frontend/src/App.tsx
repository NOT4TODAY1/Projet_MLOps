import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { InputWithFeedback } from '@/components/ui/input-with-feedback'
import { Toggle } from '@/components/ui/toggle'
import { api, PredictionInput, PredictionOutput } from '@/services/api'
import { AlertCircle, CheckCircle2, Loader2, RefreshCw, Upload, Download, Copy, History, Info, Moon, Sun, X, FileDown, FileText, Trash2 } from 'lucide-react'
import { BackgroundPathsOnly } from '@/components/ui/background-paths-only'
import { cn } from '@/lib/utils'
import { exportToPDF, exportToCSV, exportPatientDataToJSON } from '@/utils/export'
import { saveToHistory, getHistory, clearHistory, deleteHistoryItem, PredictionHistoryItem } from '@/utils/history'
import { validatePatientData, ValidationWarning } from '@/utils/validation'

const initialFormData: PredictionInput = {
  Age: 65,
  Gender: 1,
  Ethnicity: 2,
  EducationLevel: 4,
  BMI: 25.5,
  Smoking: 0,
  AlcoholConsumption: 2,
  PhysicalActivity: 3,
  DietQuality: 7,
  SleepQuality: 6,
  FamilyHistoryAlzheimers: 0,
  CardiovascularDisease: 0,
  Diabetes: 0,
  Depression: 0,
  HeadInjury: 0,
  Hypertension: 0,
  SystolicBP: 130,
  DiastolicBP: 85,
  CholesterolTotal: 200,
  CholesterolLDL: 120,
  CholesterolHDL: 50,
  CholesterolTriglycerides: 150,
  MMSE: 25,
  FunctionalAssessment: 7,
  MemoryComplaints: 0,
  BehavioralProblems: 0,
  ADL: 5,
  Confusion: 0,
  Disorientation: 0,
  PersonalityChanges: 0,
  DifficultyCompletingTasks: 0,
  Forgetfulness: 0,
}

const fieldGroups = [
  {
    title: 'Demographic Information',
    fields: [
      { key: 'Age' as keyof PredictionInput, label: 'Age (years)', type: 'number', isBinary: false },
      { key: 'Gender' as keyof PredictionInput, label: 'Gender', type: 'binary', isBinary: true, trueLabel: 'Male', falseLabel: 'Female' },
      { key: 'Ethnicity' as keyof PredictionInput, label: 'Ethnicity', type: 'number', isBinary: false },
      { key: 'EducationLevel' as keyof PredictionInput, label: 'Education Level (0-10)', type: 'number', isBinary: false },
    ],
  },
  {
    title: 'Physical Health',
    fields: [
      { key: 'BMI' as keyof PredictionInput, label: 'BMI', type: 'number', isBinary: false },
      { key: 'SystolicBP' as keyof PredictionInput, label: 'Systolic BP (mmHg)', type: 'number', isBinary: false },
      { key: 'DiastolicBP' as keyof PredictionInput, label: 'Diastolic BP (mmHg)', type: 'number', isBinary: false },
      { key: 'CholesterolTotal' as keyof PredictionInput, label: 'Total Cholesterol (mg/dL)', type: 'number', isBinary: false },
      { key: 'CholesterolLDL' as keyof PredictionInput, label: 'LDL Cholesterol (mg/dL)', type: 'number', isBinary: false },
      { key: 'CholesterolHDL' as keyof PredictionInput, label: 'HDL Cholesterol (mg/dL)', type: 'number', isBinary: false },
      { key: 'CholesterolTriglycerides' as keyof PredictionInput, label: 'Triglycerides (mg/dL)', type: 'number', isBinary: false },
    ],
  },
  {
    title: 'Lifestyle Factors',
    fields: [
      { key: 'Smoking' as keyof PredictionInput, label: 'Smoking', type: 'binary', isBinary: true },
      { key: 'AlcoholConsumption' as keyof PredictionInput, label: 'Alcohol Consumption (0-5)', type: 'number', isBinary: false },
      { key: 'PhysicalActivity' as keyof PredictionInput, label: 'Physical Activity (0-5)', type: 'number', isBinary: false },
      { key: 'DietQuality' as keyof PredictionInput, label: 'Diet Quality (0-10)', type: 'number', isBinary: false },
      { key: 'SleepQuality' as keyof PredictionInput, label: 'Sleep Quality (0-10)', type: 'number', isBinary: false },
    ],
  },
  {
    title: 'Medical History',
    fields: [
      { key: 'FamilyHistoryAlzheimers' as keyof PredictionInput, label: 'Family History of Alzheimer\'s', type: 'binary', isBinary: true },
      { key: 'CardiovascularDisease' as keyof PredictionInput, label: 'Cardiovascular Disease', type: 'binary', isBinary: true },
      { key: 'Diabetes' as keyof PredictionInput, label: 'Diabetes', type: 'binary', isBinary: true },
      { key: 'Depression' as keyof PredictionInput, label: 'Depression', type: 'binary', isBinary: true },
      { key: 'HeadInjury' as keyof PredictionInput, label: 'Head Injury', type: 'binary', isBinary: true },
      { key: 'Hypertension' as keyof PredictionInput, label: 'Hypertension', type: 'binary', isBinary: true },
    ],
  },
  {
    title: 'Cognitive Assessment',
    fields: [
      { key: 'MMSE' as keyof PredictionInput, label: 'MMSE Score (0-30)', type: 'number', isBinary: false },
      { key: 'FunctionalAssessment' as keyof PredictionInput, label: 'Functional Assessment (0-10)', type: 'number', isBinary: false },
      { key: 'ADL' as keyof PredictionInput, label: 'ADL Score (0-10)', type: 'number', isBinary: false },
    ],
  },
  {
    title: 'Symptoms',
    fields: [
      { key: 'MemoryComplaints' as keyof PredictionInput, label: 'Memory Complaints', type: 'binary', isBinary: true },
      { key: 'BehavioralProblems' as keyof PredictionInput, label: 'Behavioral Problems', type: 'binary', isBinary: true },
      { key: 'Confusion' as keyof PredictionInput, label: 'Confusion', type: 'binary', isBinary: true },
      { key: 'Disorientation' as keyof PredictionInput, label: 'Disorientation', type: 'binary', isBinary: true },
      { key: 'PersonalityChanges' as keyof PredictionInput, label: 'Personality Changes', type: 'binary', isBinary: true },
      { key: 'DifficultyCompletingTasks' as keyof PredictionInput, label: 'Difficulty Completing Tasks', type: 'binary', isBinary: true },
      { key: 'Forgetfulness' as keyof PredictionInput, label: 'Forgetfulness', type: 'binary', isBinary: true },
    ],
  },
]

// Field validation rules
const fieldValidationRules: Partial<Record<keyof PredictionInput, {
  required?: boolean
  min?: number
  max?: number
  validate?: (value: number) => string | null
}>> = {
  Age: { required: true, min: 0, max: 150 },
  Gender: { required: true, min: 0, max: 1 },
  Ethnicity: { required: true, min: 0 },
  EducationLevel: { required: true, min: 0, max: 10 },
  BMI: { required: true, min: 10, max: 50 },
  Smoking: { required: true, min: 0, max: 1 },
  AlcoholConsumption: { required: true, min: 0, max: 5 },
  PhysicalActivity: { required: true, min: 0, max: 5 },
  DietQuality: { required: true, min: 0, max: 10 },
  SleepQuality: { required: true, min: 0, max: 10 },
  FamilyHistoryAlzheimers: { required: true, min: 0, max: 1 },
  CardiovascularDisease: { required: true, min: 0, max: 1 },
  Diabetes: { required: true, min: 0, max: 1 },
  Depression: { required: true, min: 0, max: 1 },
  HeadInjury: { required: true, min: 0, max: 1 },
  Hypertension: { required: true, min: 0, max: 1 },
  SystolicBP: { required: true, min: 50, max: 250 },
  DiastolicBP: { required: true, min: 30, max: 150 },
  CholesterolTotal: { required: true, min: 0, max: 500 },
  CholesterolLDL: { required: true, min: 0, max: 300 },
  CholesterolHDL: { required: true, min: 0, max: 150 },
  CholesterolTriglycerides: { required: true, min: 0, max: 500 },
  MMSE: { required: true, min: 0, max: 30 },
  FunctionalAssessment: { required: true, min: 0, max: 10 },
  ADL: { required: true, min: 0, max: 10 },
  MemoryComplaints: { required: true, min: 0, max: 1 },
  BehavioralProblems: { required: true, min: 0, max: 1 },
  Confusion: { required: true, min: 0, max: 1 },
  Disorientation: { required: true, min: 0, max: 1 },
  PersonalityChanges: { required: true, min: 0, max: 1 },
  DifficultyCompletingTasks: { required: true, min: 0, max: 1 },
  Forgetfulness: { required: true, min: 0, max: 1 },
}

// Field tooltips
const getFieldTooltip = (key: keyof PredictionInput): string => {
  const tooltips: Partial<Record<keyof PredictionInput, string>> = {
    MMSE: 'Mini-Mental State Examination (0-30). Normal range: 24-30. Scores below 24 indicate cognitive impairment.',
    FunctionalAssessment: 'Assessment of daily functional abilities (0-10). Higher scores indicate better function.',
    ADL: 'Activities of Daily Living score (0-10). Measures independence in daily tasks.',
    Age: 'Patient age in years. Advanced age is a risk factor for Alzheimer\'s disease.',
    BMI: 'Body Mass Index. Normal range: 18.5-24.9.',
    SystolicBP: 'Systolic blood pressure in mmHg. Normal: < 120, Elevated: 120-129, High: ≥ 130.',
    DiastolicBP: 'Diastolic blood pressure in mmHg. Normal: < 80, Elevated: 80-89, High: ≥ 90.',
    MemoryComplaints: 'Patient reports memory problems (Yes/No)',
    Confusion: 'Patient experiences confusion episodes (Yes/No)',
    Disorientation: 'Patient shows disorientation (Yes/No)',
    Forgetfulness: 'Patient exhibits forgetfulness (Yes/No)',
  }
  return tooltips[key] || `Enter the ${key.replace(/([A-Z])/g, ' $1').toLowerCase()} value`
}

function App() {
  const [formData, setFormData] = useState<PredictionInput>(() => {
    // Load from auto-save if available
    try {
      const saved = localStorage.getItem('alzheimer_form_autosave')
      if (saved) {
        return JSON.parse(saved)
      }
    } catch (e) {
      console.warn('Failed to load auto-saved form data')
    }
    return initialFormData
  })
  const [errors, setErrors] = useState<Partial<Record<keyof PredictionInput, string>>>({})
  const [touched, setTouched] = useState<Partial<Record<keyof PredictionInput, boolean>>>({})
  const [prediction, setPrediction] = useState<PredictionOutput | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isRetraining, setIsRetraining] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [validationWarnings, setValidationWarnings] = useState<ValidationWarning[]>([])
  const [history, setHistory] = useState<PredictionHistoryItem[]>([])
  const [showHistory, setShowHistory] = useState(false)
  const [showModelInfo, setShowModelInfo] = useState(false)
  const [modelInfo, setModelInfo] = useState<any>(null)
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    const saved = localStorage.getItem('theme')
    return (saved as 'dark' | 'light') || 'dark'
  })
  const [autoSaveEnabled] = useState(true)

  // Initialize on mount
  useEffect(() => {
    setHistory(getHistory())
    loadModelInfo()
    setValidationWarnings(validatePatientData(formData))
    
    // Apply theme
    if (theme === 'light') {
      document.documentElement.classList.remove('dark')
    } else {
      document.documentElement.classList.add('dark')
    }
  }, [])

  // Update validation warnings when form data changes
  useEffect(() => {
    setValidationWarnings(validatePatientData(formData))
  }, [formData])

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      // Ctrl/Cmd + Enter to submit
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault()
        const form = document.querySelector('form')
        if (form) {
          form.requestSubmit()
        }
      }
      // Ctrl/Cmd + S to save
      if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault()
        exportPatientDataToJSON(formData)
      }
    }
    window.addEventListener('keydown', handleKeyPress)
    return () => window.removeEventListener('keydown', handleKeyPress)
  }, [formData])

  const loadModelInfo = async () => {
    try {
      const info = await api.getHealth()
      setModelInfo(info)
    } catch (err) {
      console.error('Failed to load model info:', err)
    }
  }

  const toggleTheme = () => {
    const newTheme = theme === 'dark' ? 'light' : 'dark'
    setTheme(newTheme)
    localStorage.setItem('theme', newTheme)
    if (newTheme === 'light') {
      document.documentElement.classList.remove('dark')
    } else {
      document.documentElement.classList.add('dark')
    }
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text).then(() => {
      alert('Copied to clipboard!')
    }).catch(err => {
      console.error('Failed to copy:', err)
    })
  }

  const handleExportPDF = () => {
    if (!prediction) return
    const timestamp = new Date().toLocaleString()
    exportToPDF(formData, prediction, timestamp)
  }

  const handleExportCSV = () => {
    if (!prediction) return
    const timestamp = new Date().toLocaleString()
    exportToCSV(formData, prediction, timestamp)
  }

  const loadFromHistory = (item: PredictionHistoryItem) => {
    setFormData(item.patientData)
    setPrediction(item.prediction)
    setShowHistory(false)
  }

  // Calculate feature importance/contribution to prediction
  const getFeatureImportance = (): Array<{ field: keyof PredictionInput; label: string; value: number; impact: 'high' | 'medium' | 'low'; reason: string }> => {
    if (!prediction) return []
    
    const importance: Array<{ field: keyof PredictionInput; label: string; value: number; impact: 'high' | 'medium' | 'low'; reason: string }> = []
    
    // Key risk factors for Alzheimer's
    const riskFactors = [
      { field: 'MMSE' as keyof PredictionInput, label: 'MMSE Score', threshold: 24, lowerIsWorse: true },
      { field: 'FunctionalAssessment' as keyof PredictionInput, label: 'Functional Assessment', threshold: 5, lowerIsWorse: true },
      { field: 'ADL' as keyof PredictionInput, label: 'ADL Score', threshold: 4, lowerIsWorse: true },
      { field: 'Age' as keyof PredictionInput, label: 'Age', threshold: 75, lowerIsWorse: false },
    ]
    
    // Symptom count
    const symptomFields: Array<keyof PredictionInput> = [
      'MemoryComplaints', 'Confusion', 'Disorientation', 'Forgetfulness',
      'DifficultyCompletingTasks', 'PersonalityChanges', 'BehavioralProblems'
    ]
    const symptomCount = symptomFields.filter(f => formData[f] === 1).length
    
    riskFactors.forEach(({ field, label, threshold, lowerIsWorse }) => {
      const value = formData[field] as number
      let impact: 'high' | 'medium' | 'low' = 'low'
      let reason = ''
      
      if (lowerIsWorse) {
        if (value < threshold * 0.7) {
          impact = 'high'
          reason = `Very low (${value}) - strong indicator of cognitive decline`
        } else if (value < threshold) {
          impact = 'medium'
          reason = `Below normal range (${value}) - may indicate early decline`
        } else {
          impact = 'low'
          reason = `Within normal range (${value})`
        }
      } else {
        if (value > threshold * 1.2) {
          impact = 'high'
          reason = `Very high (${value}) - significant risk factor`
        } else if (value > threshold) {
          impact = 'medium'
          reason = `Elevated (${value}) - moderate risk factor`
        } else {
          impact = 'low'
          reason = `Normal range (${value})`
        }
      }
      
      if (impact !== 'low' || value !== initialFormData[field]) {
        importance.push({ field, label, value, impact, reason })
      }
    })
    
    // Add symptom count
    if (symptomCount > 0) {
      importance.push({
        field: 'MemoryComplaints',
        label: 'Cognitive Symptoms',
        value: symptomCount,
        impact: symptomCount >= 5 ? 'high' : symptomCount >= 3 ? 'medium' : 'low',
        reason: `${symptomCount} symptom(s) present - ${symptomCount >= 5 ? 'highly concerning pattern' : symptomCount >= 3 ? 'moderate concern' : 'mild concern'}`
      })
    }
    
    // Sort by impact
    return importance.sort((a, b) => {
      const impactOrder = { high: 3, medium: 2, low: 1 }
      return impactOrder[b.impact] - impactOrder[a.impact]
    }).slice(0, 8) // Top 8 most important
  }

  const handleBatchUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files
    if (!files || files.length === 0) return

    setIsUploading(true)
    setError(null)

    try {
      const results: Array<{ file: string; prediction: PredictionOutput; error?: string }> = []
      
      for (const file of Array.from(files)) {
        if (file.type === 'application/json' || file.name.endsWith('.json')) {
          try {
            const patientData = await parseJSONFile(file)
            const prediction = await api.predict(patientData)
            results.push({ file: file.name, prediction })
            saveToHistory(patientData, prediction)
          } catch (err) {
            results.push({ 
              file: file.name, 
              prediction: { prediction: -1, diagnosis: 'Error' },
              error: err instanceof Error ? err.message : 'Unknown error'
            })
          }
        }
      }

      // Create CSV with batch results
      const csvRows = [['File', 'Diagnosis', 'Prediction Code', 'Confidence', 'Error']]
      results.forEach(r => {
        csvRows.push([
          r.file,
          r.prediction.diagnosis,
          String(r.prediction.prediction),
          r.prediction.probabilities 
            ? String((r.prediction.probabilities[r.prediction.prediction === 1 ? 'alzheimer' : 'healthy'] * 100).toFixed(1) + '%')
            : 'N/A',
          r.error || ''
        ])
      })

      const csvContent = csvRows.map(row => row.map(cell => `"${cell}"`).join(',')).join('\n')
      const blob = new Blob([csvContent], { type: 'text/csv' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `batch-predictions-${new Date().toISOString().split('T')[0]}.csv`
      a.click()
      URL.revokeObjectURL(url)

      alert(`Processed ${results.length} files. Results downloaded as CSV.`)
      setHistory(getHistory())
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to process batch files')
    } finally {
      setIsUploading(false)
      event.target.value = ''
    }
  }

  // Validate a single field
  const validateField = (key: keyof PredictionInput, value: number | string): string | null => {
    const rules = fieldValidationRules[key]
    if (!rules) return null

    const numValue = typeof value === 'string' ? parseFloat(value) : value

    // Check if empty and required
    if (rules.required && (value === '' || value === null || value === undefined || isNaN(numValue))) {
      return 'This field is required'
    }

    // Skip other validations if empty and not required
    if (value === '' || isNaN(numValue)) {
      return null
    }

    // Check min value
    if (rules.min !== undefined && numValue < rules.min) {
      return `Value must be at least ${rules.min}`
    }

    // Check max value
    if (rules.max !== undefined && numValue > rules.max) {
      return `Value must be at most ${rules.max}`
    }

    // Custom validation
    if (rules.validate) {
      return rules.validate(numValue)
    }

    return null
  }

  const handleInputChange = (key: keyof PredictionInput, value: string) => {
    const numValue = parseFloat(value)
    
    // Allow empty string or valid number
    if (!isNaN(numValue) || value === '') {
      const newData = { ...formData, [key]: value === '' ? 0 : numValue }
      setFormData(newData)
      
      // Auto-save to localStorage
      if (autoSaveEnabled) {
        try {
          localStorage.setItem('alzheimer_form_autosave', JSON.stringify(newData))
        } catch (e) {
          console.warn('Failed to auto-save form data')
        }
      }
      
      // Update validation warnings
      const warnings = validatePatientData(newData)
      setValidationWarnings(warnings)
      
      // Validate on change if field has been touched
      if (touched[key]) {
        const error = validateField(key, value === '' ? '' : numValue)
        setErrors(prev => {
          if (error) {
            return { ...prev, [key]: error }
          } else {
            const newErrors = { ...prev }
            delete newErrors[key]
            return newErrors
          }
        })
      }
    }
  }

  const handleBlur = (key: keyof PredictionInput) => {
    setTouched(prev => ({ ...prev, [key]: true }))
    const value = formData[key]
    const error = validateField(key, value)
    setErrors(prev => {
      if (error) {
        return { ...prev, [key]: error }
      } else {
        const newErrors = { ...prev }
        delete newErrors[key]
        return newErrors
      }
    })
  }

  const validateForm = (): boolean => {
    const newErrors: Partial<Record<keyof PredictionInput, string>> = {}
    
    Object.keys(formData).forEach(key => {
      const fieldKey = key as keyof PredictionInput
      const error = validateField(fieldKey, formData[fieldKey])
      if (error) {
        newErrors[fieldKey] = error
      }
    })

    setErrors(newErrors)
    // Mark all fields as touched
    const allTouched: Partial<Record<keyof PredictionInput, boolean>> = {}
    Object.keys(formData).forEach(key => {
      allTouched[key as keyof PredictionInput] = true
    })
    setTouched(allTouched)
    
    return Object.keys(newErrors).length === 0
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    setPrediction(null)

    if (!validateForm()) {
      return
    }

    setIsLoading(true)
    try {
      // Debug: Log what we're sending
      console.log('Submitting form data:', formData)
      console.log('Key features:', {
        MMSE: formData.MMSE,
        MemoryComplaints: formData.MemoryComplaints,
        FunctionalAssessment: formData.FunctionalAssessment,
        ADL: formData.ADL,
        Confusion: formData.Confusion,
        Disorientation: formData.Disorientation,
        Forgetfulness: formData.Forgetfulness
      })
      
      const result = await api.predict(formData)
      console.log('Prediction result:', result)
      setPrediction(result)
      
      // Save to history
      saveToHistory(formData, result)
      setHistory(getHistory())
    } catch (err) {
      console.error('Prediction error:', err)
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setIsLoading(false)
    }
  }

  const handleRetrain = async () => {
    setIsRetraining(true)
    setError(null)
    try {
      const result = await api.retrain()
      if (result.status === 'success') {
        alert('Models retrained successfully!')
      } else {
        setError(result.message)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred during retraining')
    } finally {
      setIsRetraining(false)
    }
  }

  const handleReset = () => {
    setFormData(initialFormData)
    setErrors({})
    setTouched({})
    setPrediction(null)
    setError(null)
  }

  const handleLoadSickPatientExample = () => {
    const sickPatientData: PredictionInput = {
      Age: 78,
      Gender: 1,
      Ethnicity: 2,
      EducationLevel: 3,
      BMI: 22.3,
      Smoking: 0,
      AlcoholConsumption: 1,
      PhysicalActivity: 1,
      DietQuality: 4,
      SleepQuality: 3,
      FamilyHistoryAlzheimers: 1,
      CardiovascularDisease: 1,
      Diabetes: 1,
      Depression: 1,
      HeadInjury: 0,
      Hypertension: 1,
      SystolicBP: 145,
      DiastolicBP: 92,
      CholesterolTotal: 235,
      CholesterolLDL: 155,
      CholesterolHDL: 42,
      CholesterolTriglycerides: 190,
      MMSE: 18, // LOW - concerning
      FunctionalAssessment: 3, // LOW
      MemoryComplaints: 1,
      BehavioralProblems: 1,
      ADL: 2, // LOW
      Confusion: 1,
      Disorientation: 1,
      PersonalityChanges: 1,
      DifficultyCompletingTasks: 1,
      Forgetfulness: 1,
    }
    setFormData(sickPatientData)
    setErrors({})
    setTouched({})
    setPrediction(null)
    setError(null)
  }

  const parseJSONFile = async (file: File): Promise<PredictionInput> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = (e) => {
        try {
          const jsonData = JSON.parse(e.target?.result as string)
          // Validate and map the JSON data to PredictionInput format
          const patientData: Partial<PredictionInput> = {}
          
          // Map all required fields
          Object.keys(initialFormData).forEach((key) => {
            const fieldKey = key as keyof PredictionInput
            if (jsonData[fieldKey] !== undefined) {
              patientData[fieldKey] = Number(jsonData[fieldKey])
            } else {
              // Use initial value if not provided
              patientData[fieldKey] = initialFormData[fieldKey]
            }
          })
          
          resolve(patientData as PredictionInput)
        } catch (err) {
          reject(new Error('Invalid JSON format. Please ensure the file contains valid patient data.'))
        }
      }
      reader.onerror = () => reject(new Error('Error reading file'))
      reader.readAsText(file)
    })
  }

  const parsePDFFile = async (file: File): Promise<PredictionInput> => {
    try {
      // Dynamic import of pdfjs-dist
      const pdfjsLib = await import('pdfjs-dist')
      pdfjsLib.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.js`
      
      const arrayBuffer = await file.arrayBuffer()
      const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise
      
      let fullText = ''
      for (let i = 1; i <= pdf.numPages; i++) {
        const page = await pdf.getPage(i)
        const textContent = await page.getTextContent()
        fullText += textContent.items.map((item: any) => item.str).join(' ') + '\n'
      }
      
      // Extract data from PDF text using regex patterns
      const patientData: Partial<PredictionInput> = { ...initialFormData }
      
      // Common patterns to extract from PDF
      const patterns: Record<string, RegExp> = {
        Age: /Age[:\s]+(\d+)/i,
        Gender: /Gender[:\s]+(?:Male|M|1)[:\s]*(\d+)?/i,
        MMSE: /MMSE[:\s]+(\d+)/i,
        BMI: /BMI[:\s]+([\d.]+)/i,
        SystolicBP: /Systolic[:\s]+BP[:\s]+(\d+)/i,
        DiastolicBP: /Diastolic[:\s]+BP[:\s]+(\d+)/i,
        FunctionalAssessment: /Functional[:\s]+Assessment[:\s]+(\d+)/i,
        ADL: /ADL[:\s]+(\d+)/i,
      }
      
      // Extract numeric values
      Object.keys(patterns).forEach((key) => {
        const match = fullText.match(patterns[key])
        if (match && match[1]) {
          const value = parseFloat(match[1])
          if (!isNaN(value)) {
            patientData[key as keyof PredictionInput] = value
          }
        }
      })
      
      // Extract binary fields (Yes/No, 1/0)
      const binaryPatterns: Record<string, RegExp> = {
        MemoryComplaints: /Memory[:\s]+Complaints[:\s]+(?:Yes|1|True)/i,
        Confusion: /Confusion[:\s]+(?:Yes|1|True)/i,
        Disorientation: /Disorientation[:\s]+(?:Yes|1|True)/i,
        Forgetfulness: /Forgetfulness[:\s]+(?:Yes|1|True)/i,
        Smoking: /Smoking[:\s]+(?:Yes|1|True)/i,
        Diabetes: /Diabetes[:\s]+(?:Yes|1|True)/i,
        Hypertension: /Hypertension[:\s]+(?:Yes|1|True)/i,
      }
      
      Object.keys(binaryPatterns).forEach((key) => {
        if (binaryPatterns[key].test(fullText)) {
          patientData[key as keyof PredictionInput] = 1
        }
      })
      
      return patientData as PredictionInput
    } catch (err) {
      throw new Error('Error parsing PDF. Please ensure the PDF contains structured patient data or use a JSON file instead.')
    }
  }

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    setIsUploading(true)
    setError(null)
    setPrediction(null)

    try {
      let patientData: PredictionInput

      if (file.type === 'application/json' || file.name.endsWith('.json')) {
        patientData = await parseJSONFile(file)
      } else if (file.type === 'application/pdf' || file.name.endsWith('.pdf')) {
        patientData = await parsePDFFile(file)
      } else {
        throw new Error('Unsupported file type. Please upload a JSON or PDF file.')
      }

      // Validate the loaded data
      const missingFields: string[] = []
      Object.keys(initialFormData).forEach((key) => {
        const fieldKey = key as keyof PredictionInput
        if (patientData[fieldKey] === undefined || patientData[fieldKey] === null) {
          missingFields.push(key)
        }
      })

      if (missingFields.length > 0) {
        console.warn('Some fields were not found in the file:', missingFields)
        // Fill missing fields with initial values
        missingFields.forEach((key) => {
          patientData[key as keyof PredictionInput] = initialFormData[key as keyof PredictionInput]
        })
      }

      setFormData(patientData)
      setErrors({})
      setTouched({})
      setPrediction(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load patient data from file')
    } finally {
      setIsUploading(false)
      // Reset file input
      event.target.value = ''
    }
  }

  return (
    <div className={cn("flex w-full flex-col min-h-screen bg-slate-900 dark:bg-slate-900 relative")}>
      {/* Animated Paths Background */}
      <div className="absolute inset-0 z-0">
        <BackgroundPathsOnly />
      </div>

      {/* Content Layer */}
      <div className="relative z-10 flex flex-col flex-1 py-8 px-4">
        <div className="max-w-7xl mx-auto w-full">
          {/* Header */}
          <div className="mb-8">
            <div className="flex items-center justify-between mb-4">
              <div className="flex gap-2">
                <Button
                  type="button"
                  variant="outline"
                  onClick={toggleTheme}
                  className="border-white/20 bg-transparent text-white hover:bg-white/10"
                  title="Toggle theme"
                >
                  {theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setShowModelInfo(!showModelInfo)}
                  className="border-white/20 bg-transparent text-white hover:bg-white/10"
                  title="Model Information"
                >
                  <Info className="h-4 w-4 mr-2" />
                  Model Info
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => { setShowHistory(!showHistory); setHistory(getHistory()) }}
                  className="border-white/20 bg-transparent text-white hover:bg-white/10"
                  title="View prediction history"
                >
                  <History className="h-4 w-4 mr-2" />
                  History ({history.length})
                </Button>
              </div>
            </div>
            <div className="text-center">
              <h1 className="text-4xl font-bold text-white mb-2 dark:text-white">
                Alzheimer's Disease Classifier
              </h1>
              <p className="text-lg text-white/70 dark:text-white/70">
                Machine Learning API for Alzheimer's Disease Classification
              </p>
            </div>
          </div>

          {/* Validation Warnings */}
          {validationWarnings.length > 0 && (
            <Card className="backdrop-blur-sm bg-yellow-500/10 border-yellow-500/30 mb-6">
              <CardContent className="pt-6">
                <div className="space-y-2">
                  <div className="flex items-center gap-2 text-yellow-300 font-semibold mb-2">
                    <AlertCircle className="h-5 w-5" />
                    Data Validation Warnings
                  </div>
                  {validationWarnings.map((warning, idx) => (
                    <div
                      key={idx}
                      className={cn(
                        "p-3 rounded text-sm",
                        warning.severity === 'high' && "bg-red-500/10 border border-red-500/30 text-red-300",
                        warning.severity === 'medium' && "bg-yellow-500/10 border border-yellow-500/30 text-yellow-300",
                        warning.severity === 'low' && "bg-blue-500/10 border border-blue-500/30 text-blue-300"
                      )}
                    >
                      <strong>{warning.field}:</strong> {warning.message}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Model Info Panel */}
          {showModelInfo && modelInfo && (
            <Card className="backdrop-blur-sm bg-white/5 border-white/10 mb-6">
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="text-white">Model Information</CardTitle>
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => setShowModelInfo(false)}
                  className="border-white/20 bg-transparent text-white hover:bg-white/10"
                >
                  <X className="h-4 w-4" />
                </Button>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-white/80 text-sm">
                  <p><strong>Status:</strong> {modelInfo.status}</p>
                  <p><strong>Model Type:</strong> {modelInfo.model_type || 'Unknown'}</p>
                  <p><strong>Model Loaded:</strong> {modelInfo.model_loaded ? 'Yes' : 'No'}</p>
                  <p><strong>Scaler Loaded:</strong> {modelInfo.scaler_loaded ? 'Yes' : 'No'}</p>
                  <p><strong>Model Path:</strong> {modelInfo.model_path}</p>
                </div>
              </CardContent>
            </Card>
          )}

          {/* History Panel */}
          {showHistory && (
            <Card className="backdrop-blur-sm bg-white/5 border-white/10 mb-6">
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="text-white">Prediction History</CardTitle>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      if (confirm('Clear all history?')) {
                        clearHistory()
                        setHistory([])
                      }
                    }}
                    className="border-white/20 bg-transparent text-white hover:bg-white/10"
                  >
                    Clear All
                  </Button>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() => setShowHistory(false)}
                    className="border-white/20 bg-transparent text-white hover:bg-white/10"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                {history.length === 0 ? (
                  <p className="text-white/60 text-center py-4">No prediction history yet</p>
                ) : (
                  <div className="space-y-2 max-h-96 overflow-y-auto">
                    {history.map((item) => (
                      <div
                        key={item.id}
                        className="p-3 bg-white/5 rounded border border-white/10 hover:bg-white/10 transition"
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex-1">
                            <p className="text-white font-medium">
                              {item.prediction.diagnosis} - {new Date(item.timestamp).toLocaleString()}
                            </p>
                            {item.prediction.probabilities && (
                              <p className="text-white/60 text-sm">
                                Confidence: {(item.prediction.probabilities[item.prediction.prediction === 1 ? 'alzheimer' : 'healthy'] * 100).toFixed(1)}%
                              </p>
                            )}
                          </div>
                          <div className="flex gap-2">
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => loadFromHistory(item)}
                              className="border-white/20 bg-transparent text-white hover:bg-white/10"
                            >
                              Load
                            </Button>
                            <Button
                              variant="outline"
                              size="icon"
                              onClick={() => {
                                deleteHistoryItem(item.id)
                                setHistory(getHistory())
                              }}
                              className="border-white/20 bg-transparent text-white hover:bg-white/10"
                            >
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* Form Section - Landscape Layout */}
          <Card className="backdrop-blur-sm bg-white/5 border-white/10">
            <CardHeader>
              <CardTitle className="text-white">Patient Information</CardTitle>
              <CardDescription className="text-white/60">
                Enter patient clinical and demographic data for prediction
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-8">
                {fieldGroups.map((group, groupIndex) => (
                  <div key={groupIndex} className="space-y-4">
                    <h3 className="text-lg font-semibold text-white border-b border-white/10 pb-2">
                      {group.title}
                    </h3>
                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
                      {group.fields.map((field) => (
                        <div key={field.key} className="space-y-2">
                          {field.isBinary ? (
                            <Toggle
                              id={field.key}
                              label={field.label}
                              checked={formData[field.key] === 1}
                              onChange={(checked) => {
                                setFormData(prev => ({ ...prev, [field.key]: checked ? 1 : 0 }))
                                if (touched[field.key]) {
                                  const error = validateField(field.key, checked ? 1 : 0)
                                  setErrors(prev => {
                                    if (error) {
                                      return { ...prev, [field.key]: error }
                                    } else {
                                      const newErrors = { ...prev }
                                      delete newErrors[field.key]
                                      return newErrors
                                    }
                                  })
                                }
                              }}
                              onBlur={() => handleBlur(field.key)}
                              trueLabel={field.trueLabel || 'Yes'}
                              falseLabel={field.falseLabel || 'No'}
                            />
                          ) : (
                            <>
                              <div className="flex items-center gap-1">
                                <Label htmlFor={field.key} className="text-white/80 text-sm">
                                  {field.label}
                                </Label>
                                <div title={getFieldTooltip(field.key)}>
                                  <Info className="h-3 w-3 text-white/40 cursor-help" />
                                </div>
                              </div>
                              <InputWithFeedback
                                id={field.key}
                                type={field.type}
                                value={formData[field.key] || ''}
                                onChange={(e) => handleInputChange(field.key, e.target.value)}
                                onBlur={() => handleBlur(field.key)}
                                isError={!!errors[field.key] && !!touched[field.key]}
                                errorMessage={touched[field.key] ? errors[field.key] : undefined}
                                className="bg-white/5 border-white/10 text-white placeholder:text-white/40 focus-visible:ring-white/20"
                              />
                            </>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                ))}

                <div className="flex flex-wrap gap-4 pt-4 border-t border-white/10">
                  <Button 
                    type="submit" 
                    disabled={isLoading || Object.keys(errors).length > 0} 
                    className="flex-1 bg-white text-black hover:bg-white/90 font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isLoading ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Predicting...
                      </>
                    ) : (
                      'Predict'
                    )}
                  </Button>
                  <Button
                    type="button"
                    variant="outline"
                    onClick={handleReset}
                    disabled={isLoading || isUploading}
                    className="border-white/20 bg-transparent text-white hover:bg-white/10 hover:text-white disabled:opacity-50"
                  >
                    Reset
                  </Button>
                  <div className="relative">
                    <input
                      type="file"
                      accept=".json,.pdf,application/json,application/pdf"
                      onChange={handleFileUpload}
                      disabled={isLoading || isUploading}
                      multiple={false}
                      className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
                      id="file-upload"
                    />
                    <input
                      type="file"
                      accept=".json,application/json"
                      onChange={handleBatchUpload}
                      disabled={isLoading || isUploading}
                      multiple={true}
                      className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
                      id="batch-upload"
                      style={{ display: 'none' }}
                    />
                    <Button
                      type="button"
                      variant="outline"
                      disabled={isLoading || isUploading}
                      className="border-blue-500/30 text-blue-300 hover:bg-blue-500/10 hover:border-blue-500/50"
                      title="Upload JSON or PDF file with patient data"
                      asChild
                    >
                      <label htmlFor="file-upload" className="cursor-pointer">
                        {isUploading ? (
                          <>
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            Uploading...
                          </>
                        ) : (
                          <>
                            <Upload className="mr-2 h-4 w-4" />
                            Upload File
                          </>
                        )}
                      </label>
                    </Button>
                  </div>
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => document.getElementById('batch-upload')?.click()}
                    disabled={isLoading || isUploading}
                    className="border-indigo-500/30 text-indigo-300 hover:bg-indigo-500/10 hover:border-indigo-500/50"
                    title="Upload multiple JSON files for batch processing"
                  >
                    <Upload className="mr-2 h-4 w-4" />
                    Batch Process
                  </Button>
                  <Button
                    type="button"
                    variant="outline"
                    onClick={handleLoadSickPatientExample}
                    disabled={isLoading || isUploading}
                    className="border-red-500/30 text-red-300 hover:bg-red-500/10 hover:border-red-500/50"
                    title="Load example patient with Alzheimer's symptoms"
                  >
                    Load Sick Patient Example
                  </Button>
                  <Button
                    type="button"
                    variant="outline"
                    onClick={handleRetrain}
                    disabled={isRetraining || isLoading || isUploading}
                    className="border-white/20 bg-transparent text-white hover:bg-white/10 hover:text-white disabled:opacity-50"
                  >
                    {isRetraining ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Retraining...
                      </>
                    ) : (
                      <>
                        <RefreshCw className="mr-2 h-4 w-4" />
                        Retrain Models
                      </>
                    )}
                  </Button>
                  {prediction && (
                    <>
                      <Button
                        type="button"
                        variant="outline"
                        onClick={handleExportPDF}
                        className="border-green-500/30 text-green-300 hover:bg-green-500/10 hover:border-green-500/50"
                        title="Export as PDF"
                      >
                        <FileText className="mr-2 h-4 w-4" />
                        Export PDF
                      </Button>
                      <Button
                        type="button"
                        variant="outline"
                        onClick={handleExportCSV}
                        className="border-blue-500/30 text-blue-300 hover:bg-blue-500/10 hover:border-blue-500/50"
                        title="Export as CSV"
                      >
                        <FileDown className="mr-2 h-4 w-4" />
                        Export CSV
                      </Button>
                      <Button
                        type="button"
                        variant="outline"
                        onClick={() => exportPatientDataToJSON(formData)}
                        className="border-purple-500/30 text-purple-300 hover:bg-purple-500/10 hover:border-purple-500/50"
                        title="Export patient data as JSON"
                      >
                        <Download className="mr-2 h-4 w-4" />
                        Export JSON
                      </Button>
                    </>
                  )}
                </div>
              </form>
            </CardContent>
          </Card>

          {/* Results Section - At the Bottom */}
          {(prediction || error) && (
            <Card className="backdrop-blur-sm bg-white/5 border-white/10 mt-6">
              <CardHeader>
                <CardTitle className="text-white">Prediction Result</CardTitle>
                <CardDescription className="text-white/60">
                  Model prediction output
                </CardDescription>
              </CardHeader>
              <CardContent>
                {error && (
                  <div className="mb-4 p-4 bg-red-500/10 border border-red-500/30 rounded-md">
                    <div className="flex items-center gap-2 text-red-400">
                      <AlertCircle className="h-5 w-5" />
                      <p className="font-medium">Error</p>
                    </div>
                    <p className="mt-2 text-sm text-red-300">{error}</p>
                  </div>
                )}

                {prediction && (
                  <div className="space-y-4">
                    <div
                      className={cn(
                        "p-6 rounded-lg border-2",
                        prediction.prediction === 1
                          ? 'bg-red-500/10 border-red-500/30'
                          : 'bg-green-500/10 border-green-500/30'
                      )}
                    >
                      <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-3">
                          {prediction.prediction === 1 ? (
                            <AlertCircle className="h-8 w-8 text-red-400" />
                          ) : (
                            <CheckCircle2 className="h-8 w-8 text-green-400" />
                          )}
                          <div>
                            <h3 className="text-2xl font-bold text-white">
                              {prediction.diagnosis}
                            </h3>
                            <p className="text-sm text-white/70">
                              Prediction Code: {prediction.prediction}
                              {prediction.prediction === 0 && ' (Healthy)'}
                              {prediction.prediction === 1 && ' (Alzheimer\'s Disease)'}
                            </p>
                          </div>
                        </div>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => copyToClipboard(JSON.stringify({ prediction, formData }, null, 2))}
                          className="border-white/20 bg-transparent text-white hover:bg-white/10"
                        >
                          <Copy className="h-4 w-4 mr-2" />
                          Copy
                        </Button>
                      </div>

                      {/* Confidence/Probability Display */}
                      {prediction.probabilities && (
                        <div className="mb-4 space-y-3">
                          <div>
                            <div className="flex justify-between text-sm mb-1">
                              <span className="text-white/80">Confidence Level</span>
                              <span className="text-white font-semibold">
                                {(prediction.probabilities[prediction.prediction === 1 ? 'alzheimer' : 'healthy'] * 100).toFixed(1)}%
                              </span>
                            </div>
                            <div className="w-full bg-white/10 rounded-full h-3 overflow-hidden">
                              <div
                                className={cn(
                                  "h-full transition-all duration-500",
                                  prediction.prediction === 1 ? 'bg-red-500' : 'bg-green-500'
                                )}
                                style={{
                                  width: `${prediction.probabilities[prediction.prediction === 1 ? 'alzheimer' : 'healthy'] * 100}%`
                                }}
                              />
                            </div>
                          </div>
                          <div className="grid grid-cols-2 gap-4 text-sm">
                            <div className="p-3 bg-green-500/10 border border-green-500/30 rounded">
                              <div className="text-green-300 font-semibold">Healthy</div>
                              <div className="text-green-200 text-lg">
                                {(prediction.probabilities.healthy * 100).toFixed(1)}%
                              </div>
                            </div>
                            <div className="p-3 bg-red-500/10 border border-red-500/30 rounded">
                              <div className="text-red-300 font-semibold">Alzheimer's</div>
                              <div className="text-red-200 text-lg">
                                {(prediction.probabilities.alzheimer * 100).toFixed(1)}%
                              </div>
                            </div>
                          </div>
                        </div>
                      )}

                      {prediction.prediction === 0 && (
                        <div className="mt-4 p-3 bg-blue-500/10 border border-blue-500/30 rounded text-sm text-blue-300">
                          <p className="font-semibold mb-1">Why "Healthy"?</p>
                          <p className="text-blue-200/80">
                            The model predicts Healthy when cognitive scores (MMSE, FunctionalAssessment, ADL) are good 
                            and few symptoms are present. To get an Alzheimer's prediction, try:
                          </p>
                          <ul className="list-disc list-inside mt-2 space-y-1 text-blue-200/70">
                            <li>Lower MMSE score (&lt; 24, ideally &lt; 20)</li>
                            <li>Lower FunctionalAssessment (&lt; 5)</li>
                            <li>Lower ADL score (&lt; 4)</li>
                            <li>Multiple symptoms (Confusion, Disorientation, Forgetfulness, etc.)</li>
                          </ul>
                          <p className="mt-2 text-blue-200/80">
                            Click "Load Sick Patient Example" to see a case that predicts Alzheimer's.
                          </p>
                        </div>
                      )}

                      {/* Feature Importance */}
                      <div className="mt-6 p-4 bg-white/5 border border-white/10 rounded">
                        <h4 className="text-white font-semibold mb-3 flex items-center gap-2">
                          <Info className="h-4 w-4" />
                          Key Factors Influencing Prediction
                        </h4>
                        <div className="space-y-2">
                          {getFeatureImportance().map((item, idx) => (
                            <div
                              key={idx}
                              className={cn(
                                "p-2 rounded text-sm",
                                item.impact === 'high' && "bg-red-500/10 border border-red-500/30",
                                item.impact === 'medium' && "bg-yellow-500/10 border border-yellow-500/30",
                                item.impact === 'low' && "bg-blue-500/10 border border-blue-500/30"
                              )}
                            >
                              <div className="flex items-center justify-between">
                                <span className="text-white font-medium">{item.label}:</span>
                                <span className={cn(
                                  "text-xs px-2 py-1 rounded",
                                  item.impact === 'high' && "bg-red-500/20 text-red-300",
                                  item.impact === 'medium' && "bg-yellow-500/20 text-yellow-300",
                                  item.impact === 'low' && "bg-blue-500/20 text-blue-300"
                                )}>
                                  {item.impact.toUpperCase()}
                                </span>
                              </div>
                              <p className="text-white/70 text-xs mt-1">{item.reason}</p>
                            </div>
                          ))}
                          {getFeatureImportance().length === 0 && (
                            <p className="text-white/60 text-sm text-center py-2">
                              All values are within normal ranges
                            </p>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}

export default App

