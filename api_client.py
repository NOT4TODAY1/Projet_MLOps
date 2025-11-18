"""
Test script for the FastAPI Alzheimer's Disease Classifier.

Usage:
    python api_client.py

This script tests the /predict and /retrain endpoints.
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint."""
    print("\n=== Testing /health ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_predict():
    """Test the prediction endpoint with sample data."""
    print("\n=== Testing /predict ===")
    
    sample_data = {
        "Age": 65,
        "Gender": 1,
        "Ethnicity": 2,
        "EducationLevel": 4,
        "BMI": 25.5,
        "Smoking": 0,
        "AlcoholConsumption": 2,
        "PhysicalActivity": 3,
        "DietQuality": 7,
        "SleepQuality": 6,
        "FamilyHistoryAlzheimers": 0,
        "CardiovascularDisease": 0,
        "Diabetes": 0,
        "Depression": 0,
        "HeadInjury": 0,
        "Hypertension": 0,
        "SystolicBP": 130,
        "DiastolicBP": 85,
        "CholesterolTotal": 200,
        "CholesterolLDL": 120,
        "CholesterolHDL": 50,
        "CholesterolTriglycerides": 150,
        "MMSE": 25,
        "FunctionalAssessment": 7,
        "MemoryComplaints": 0,
        "BehavioralProblems": 0,
        "ADL": 5,
        "Confusion": 0,
        "Disorientation": 0,
        "PersonalityChanges": 0,
        "DifficultyCompletingTasks": 0,
        "Forgetfulness": 0
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=sample_data)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Error Response: {response.text}")


def test_retrain():
    """Test the retrain endpoint."""
    print("\n=== Testing /retrain ===")
    print("This will retrain all models (may take a few minutes)...")
    
    response = requests.post(f"{BASE_URL}/retrain")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Error Response: {response.text}")


if __name__ == "__main__":
    try:
        test_health()
        test_predict()
        print("\n✓ API is working!")
        print("\nTo test /retrain, uncomment test_retrain() call (takes several minutes)")
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to API at {BASE_URL}")
        print("Make sure the server is running: python -m uvicorn app:app --reload")

