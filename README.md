# CampusWell – Student Wellbeing Predictor

A machine learning web application that predicts depression risk for college students using Logistic Regression and provides personalized, actionable remedies. Built with FastAPI (backend) and vanilla HTML/CSS/JavaScript (frontend).

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Features](#features)
- [API Endpoints](#api-endpoints)
- [Model Information](#model-information)

## 🎯 Project Overview

This project helps college students assess their potential depression risk by analyzing factors such as:
- Course failures
- College absences
- Social activity (going out frequency)
- Weekly study hours
- Health status
- GPA
- Alcohol consumption habits

The system uses a Logistic Regression model trained on student data and compares it with other algorithms (Naive Bayes, SVM, KNN) to demonstrate performance.

## ✅ Prerequisites

Before running this project, ensure you have:

1. **Python 3.10 or higher** installed
   - Check version: `python --version` or `py --version`
   - Download from: https://www.python.org/downloads/

2. **pip** (Python package manager) - usually comes with Python
   - Check: `pip --version`

3. **A modern web browser** (Chrome, Firefox, Edge, etc.)

4. **The dataset file**: `studataset.csv` should be in the project root directory

## 📦 Installation

### Step 1: Navigate to Project Directory

Open PowerShell (Windows) or Terminal (Mac/Linux) and navigate to the project folder:

```powershell
cd "D:\study college\3rd year\ML-PROJECT"
```

### Step 2: Install Python Dependencies

Install all required packages:

```powershell
pip install -r requirements.txt
```

**Note**: If `pip` doesn't work, try `py -m pip install -r requirements.txt` on Windows.

This will install:
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning library
- `fastapi` - Web framework for API
- `uvicorn` - ASGI server

### Step 3: (Optional) Train/Evaluate Models

The model (`backend/model.pkl`) is already included. If you want to retrain or evaluate all models:

```powershell
python backend/evaluate_models.py
```

This will:
- Train 4 models: Logistic Regression, Naive Bayes, SVM, KNN
- Calculate test accuracies
- Save metrics to `backend/metrics.json`
- Update `backend/model.pkl` with the Logistic Regression model

## 🚀 How to Run

### Running the Backend (API Server)

1. **Open a PowerShell/Terminal window**

2. **Navigate to project root**:
   ```powershell
   cd "D:\study college\3rd year\ML-PROJECT"
   ```

3. **Start the FastAPI server**:
   ```powershell
   uvicorn backend.main:app --reload
   ```

4. **Verify it's running**: You should see:
   ```
   INFO:     Uvicorn running on http://127.0.0.1:8000
   ```

5. **Test the API**: Open http://127.0.0.1:8000/docs in your browser to see the interactive API documentation.

**Important**: Keep this terminal window open while using the application.

### Running the Frontend (Web Interface)

1. **Open a NEW PowerShell/Terminal window** (keep the backend running in the first one)

2. **Navigate to the frontend folder**:
   ```powershell
   cd "D:\study college\3rd year\ML-PROJECT\frontend"
   ```

3. **Start a local web server**:
   ```powershell
   python -m http.server 5501
   ```
   
   **Alternative on Windows**:
   ```powershell
   py -m http.server 5501
   ```

4. **Open the application**: In your browser, go to:
   ```
   http://127.0.0.1:5501/index.html
   ```

5. **You're ready!** You should see the CampusWell homepage.

## 📁 Project Structure

```
ML-PROJECT/
│
├── backend/                 # Backend API code
│   ├── main.py             # FastAPI application & endpoints
│   ├── train_model.py      # Model training script
│   ├── evaluate_models.py  # Model comparison script
│   ├── model.pkl           # Trained Logistic Regression model
│   └── metrics.json        # Model accuracy metrics
│
├── frontend/                # Frontend web files
│   ├── index.html          # Homepage
│   ├── predict.html         # Prediction form page
│   ├── result.html          # Results display page
│   ├── how.html            # How it works page
│   ├── app.js              # JavaScript logic
│   └── style.css           # Styling
│
├── studataset.csv          # Student dataset (required)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## ✨ Features

- **Multi-page website**: Homepage, Prediction form, Results page, How it works page
- **Real-time prediction**: Submit form data and get instant results
- **Personalized advice**: Receive tailored recommendations based on your inputs
- **Model comparison**: Visual chart comparing Logistic Regression with other algorithms
- **Responsive design**: Works on desktop and mobile devices
- **Privacy-first**: No data is stored on the server

## 🔌 API Endpoints

### POST `/predict`
Predicts depression risk based on student features.

**Request Body** (JSON):
```json
{
  "failures": 0,
  "absences": 5,
  "goout": 3,
  "alcohol_habit": false,
  "studytime": "4-8",
  "health": "good",
  "G3": 8.5,
  "sex": "M",
  "age": 20
}
```

**Response**:
```json
{
  "prediction": 0,
  "advice": [
    "Maintain your healthy habits...",
    "Great job keeping alcohol at 0..."
  ]
}
```

- `prediction`: `0` = Low risk, `1` = High risk
- `advice`: Array of personalized recommendations

### GET `/metrics`
Returns model accuracy metrics for comparison chart.

**Response**:
```json
{
  "logistic_regression": 0.85,
  "naive_bayes": 0.78,
  "svm_rbf": 0.84,
  "knn": 0.80
}
```

## 🤖 Model Information

### Current Model: Logistic Regression
- **Accuracy**: ~85% (may vary based on dataset)
- **Features Used**: 8 features (failures, absences, goout, Dalc, Walc, studytime, health, G3)
- **Why Logistic Regression**: Provides good accuracy with interpretability, making it suitable for this application.

### Model Comparison
The project compares Logistic Regression with:
1. **Naive Bayes** - Lower accuracy (~78%), probabilistic approach
2. **SVM (RBF)** - Similar accuracy (~84%), kernel-based
3. **KNN** - Lower accuracy (~80%), instance-based learning

Comparison chart is available on the "How it works" page.

## 🐛 Troubleshooting

### Backend won't start
- **Error**: `ModuleNotFoundError: No module named 'backend'`
  - **Solution**: Make sure you're running `uvicorn backend.main:app --reload` from the **project root**, not from inside the `backend` folder.

### Frontend can't connect to API
- **Error**: "Error: Could not connect to the server"
  - **Solution**: Ensure the backend is running on `http://127.0.0.1:8000` and check that the port matches in `frontend/app.js`.

### Port already in use
- **Error**: `Address already in use`
  - **Solution**: Use a different port:
    - Backend: `uvicorn backend.main:app --reload --port 8001`
    - Frontend: `python -m http.server 5502`

### Model file not found
- **Error**: `FileNotFoundError: model.pkl`
  - **Solution**: Run `python backend/evaluate_models.py` to generate the model file.

## 📝 Notes

- The prediction is **for guidance only** and is **not a clinical diagnosis**
- Dataset: `studataset.csv` must be present in the project root
- All model comparisons use an 80/20 train-test split
- The application runs locally and does not require internet connection (except for loading Chart.js library)

## 👨‍🏫 For Teachers/Evaluators

**Quick Start**:
1. Install dependencies: `pip install -r requirements.txt`
2. Run backend: `uvicorn backend.main:app --reload` (from project root)
3. Run frontend: `python -m http.server 5501` (from frontend folder)
4. Open: http://127.0.0.1:5501/index.html

**Project demonstrates**:
- Machine learning model deployment (Logistic Regression)
- RESTful API development (FastAPI)
- Frontend-backend integration
- Model comparison and evaluation
- User-friendly web interface

---

**Developed by**
- M.Samsrith
- P.Sarath Chandra
- G.Kiran Kriteen

**Date**: 2025  
**Academic Project**: ML-PROJECT

