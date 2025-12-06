import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';
import { PredictionInput, PredictionOutput } from '@/services/api';

export const exportToPDF = (
  patientData: PredictionInput,
  prediction: PredictionOutput,
  timestamp: string
) => {
  const doc = new jsPDF();
  
  // Title
  doc.setFontSize(18);
  doc.text('Alzheimer\'s Disease Assessment Report', 14, 20);
  
  // Date
  doc.setFontSize(10);
  doc.text(`Generated: ${timestamp}`, 14, 30);
  
  // Prediction Result
  doc.setFontSize(14);
  if (prediction.prediction === 1) {
    doc.setFillColor(255, 0, 0); // Red for Alzheimer's
  } else {
    doc.setFillColor(0, 128, 0); // Green for Healthy
  }
  doc.setTextColor(255, 255, 255);
  doc.rect(14, 40, 180, 15, 'F');
  doc.text(`Diagnosis: ${prediction.diagnosis}`, 20, 50);
  
  // Probabilities
  if (prediction.probabilities) {
    doc.setTextColor(0, 0, 0);
    doc.setFontSize(12);
    doc.text(`Confidence: ${(prediction.probabilities[prediction.prediction === 1 ? 'alzheimer' : 'healthy'] * 100).toFixed(1)}%`, 14, 65);
  }
  
  // Patient Data Table
  doc.setFontSize(12);
  doc.text('Patient Information', 14, 80);
  
  const tableData = Object.entries(patientData).map(([key, value]) => [
    key.replace(/([A-Z])/g, ' $1').trim(),
    String(value)
  ]);
  
  autoTable(doc, {
    startY: 85,
    head: [['Field', 'Value']],
    body: tableData,
    theme: 'striped',
    headStyles: { fillColor: [66, 139, 202] },
  });
  
  // Footer
  const pageCount = doc.getNumberOfPages();
  for (let i = 1; i <= pageCount; i++) {
    doc.setPage(i);
    doc.setFontSize(8);
    doc.text(
      `Page ${i} of ${pageCount}`,
      doc.internal.pageSize.width / 2,
      doc.internal.pageSize.height - 10,
      { align: 'center' }
    );
  }
  
  doc.save(`alzheimer-assessment-${timestamp.replace(/[: ]/g, '-')}.pdf`);
};

export const exportToCSV = (
  patientData: PredictionInput,
  prediction: PredictionOutput,
  timestamp: string
) => {
  const csvRows = [
    ['Field', 'Value'],
    ['Timestamp', timestamp],
    ['Prediction', prediction.diagnosis],
    ['Prediction Code', String(prediction.prediction)],
  ];
  
  if (prediction.probabilities) {
    csvRows.push(['Confidence (Healthy)', String(prediction.probabilities.healthy)]);
    csvRows.push(['Confidence (Alzheimer)', String(prediction.probabilities.alzheimer)]);
  }
  
  csvRows.push(['', '']); // Empty row
  csvRows.push(['Patient Data', '']);
  
  Object.entries(patientData).forEach(([key, value]) => {
    csvRows.push([key, String(value)]);
  });
  
  const csvContent = csvRows.map(row => row.map(cell => `"${cell}"`).join(',')).join('\n');
  const blob = new Blob([csvContent], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `alzheimer-assessment-${timestamp.replace(/[: ]/g, '-')}.csv`;
  a.click();
  URL.revokeObjectURL(url);
};

export const exportPatientDataToJSON = (patientData: PredictionInput) => {
  const jsonContent = JSON.stringify(patientData, null, 2);
  const blob = new Blob([jsonContent], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `patient-data-${new Date().toISOString().split('T')[0]}.json`;
  a.click();
  URL.revokeObjectURL(url);
};

