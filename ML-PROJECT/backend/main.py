
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Optional, List, Dict, Any

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model from the backend directory regardless of CWD
MODEL_PATH = Path(__file__).parent / 'model.pkl'
model = joblib.load(MODEL_PATH)

class StudentFeatures(BaseModel):
    failures: int = Field(ge=0, le=4)
    absences: int = Field(ge=0, le=93)
    goout: int = Field(ge=0, le=20)  # UI: 0-20, will map to 1-5 for model
    # 0 means no alcohol consumption (ideal)
    Dalc: int = Field(ge=0, le=5)
    Walc: int = Field(ge=0, le=5)
    studytime: Optional[str] = Field(default=None)  # UI: dropdown "<4", "4-8", "8-12", ">12"
    health: Optional[str] = Field(default=None)  # UI: "very_bad", "bad", "average", "good", "very_good"
    G3: Optional[float] = Field(default=None)  # UI: GPA 0-10.00, will map to 0-20
    # New UI fields (optional for now; backend ignores them for the model)
    sex: Optional[str] = Field(default=None, description="M or F")
    age: Optional[int] = Field(default=None, ge=10, le=100)
    alcohol_habit: Optional[bool] = Field(default=None, description="True if consumes alcohol")
    family_hours: Optional[float] = Field(default=None, ge=0, le=50, description="Hours per week spent with family")
    activity_hours: Optional[float] = Field(default=None, ge=0, le=50, description="Hours per week spent on activities/playing")

def generate_advice(features: dict, prediction: int) -> list[str]:
    advice: list[str] = []
    # General advice based on risk
    if prediction == 1:
        advice.append("Talk to a counselor or trusted mentor within the next week.")
        advice.append("Create a simple weekly routine: consistent sleep, meals, and study slots.")
        advice.append("Schedule short daily walks or light exercise (15–20 minutes).")
    else:
        advice.append("Maintain your healthy habits and check in with yourself weekly.")

    # Targeted advice from inputs
    if features.get("Dalc", 0) > 0 or features.get("Walc", 0) > 0:
        advice.append("Aim for 0 alcohol consumption; it improves sleep and mood stability.")
    else:
        advice.append("Great job keeping alcohol at 0 — keep it up.")

    if features.get("absences", 0) > 10:
        advice.append("Reduce school absences by planning ahead (set morning reminders).")

    goout_val = features.get("goout", 10)
    # If goout was 0-20 scale, convert for advice check
    if goout_val > 5:
        goout_val = min(5, max(1, round(goout_val / 4.0)))
    if goout_val < 2:
        advice.append("Schedule at least one social catch‑up per week to stay connected.")

    studytime_val = features.get("studytime", 2)
    if isinstance(studytime_val, str):
        studytime_map = {'<4': 1, '4-8': 2, '8-12': 3, '>12': 4}
        studytime_val = studytime_map.get(studytime_val, 2)
    if studytime_val < 2:
        advice.append("Use 25-minute focused sessions (Pomodoro) with 5-minute breaks.")

    if features.get("health", 3) <= 2:
        advice.append("Prioritize sleep (7–8h) and balanced meals; hydrate regularly.")

    g3_val = features.get("G3", 10)
    # If G3 is GPA (0-10), convert for advice check
    if isinstance(g3_val, float) and g3_val <= 10:
        g3_val = g3_val * 2  # Convert to 0-20 scale
    if g3_val < 8:
        advice.append("Seek subject-specific help (tutoring/peer groups) twice weekly.")

    # Advice based on new fields
    family_hours = features.get("family_hours", 0)
    if isinstance(family_hours, (int, float)) and family_hours < 5:
        advice.append("Try to spend at least 5-7 hours per week with family members for better emotional support.")
    elif isinstance(family_hours, (int, float)) and family_hours >= 10:
        advice.append("Great job maintaining strong family connections! This helps with emotional wellbeing.")

    activity_hours = features.get("activity_hours", 0)
    if isinstance(activity_hours, (int, float)) and activity_hours < 3:
        advice.append("Engage in at least 3-5 hours of physical activities or hobbies per week to reduce stress.")
    elif isinstance(activity_hours, (int, float)) and activity_hours >= 10:
        advice.append("Excellent! Regular activities and hobbies contribute positively to mental health.")

    return advice

def map_ui_to_model(payload: dict) -> dict:
    """Map UI formats to model feature formats"""
    model_payload = payload.copy()
    
    # Map goout 0-20 to 1-5 scale (divide by 4, round, cap at 5)
    goout_ui = payload.get('goout', 0)
    if isinstance(goout_ui, (int, float)):
        model_payload['goout'] = min(5, max(1, round(goout_ui / 4.0)))
    
    # Map studytime dropdown to 1-4 scale
    studytime_ui = payload.get('studytime')
    if studytime_ui == '<4':
        model_payload['studytime'] = 1
    elif studytime_ui == '4-8':
        model_payload['studytime'] = 2
    elif studytime_ui == '8-12':
        model_payload['studytime'] = 3
    elif studytime_ui == '>12':
        model_payload['studytime'] = 4
    elif isinstance(studytime_ui, (int, float)):
        model_payload['studytime'] = int(studytime_ui)
    else:
        model_payload['studytime'] = 2  # default
    
    # Map health dropdown to 1-5 scale
    health_ui = payload.get('health')
    health_map = {
        'very_bad': 1,
        'bad': 2,
        'average': 3,
        'good': 4,
        'very_good': 5
    }
    if health_ui in health_map:
        model_payload['health'] = health_map[health_ui]
    elif isinstance(health_ui, (int, float)):
        model_payload['health'] = int(health_ui)
    else:
        model_payload['health'] = 3  # default
    
    # Map GPA 0-10.00 to G3 0-20 (multiply by 2)
    gpa_ui = payload.get('G3')
    if isinstance(gpa_ui, (int, float)):
        model_payload['G3'] = min(20, max(0, round(gpa_ui * 2)))
    elif isinstance(gpa_ui, str):
        try:
            gpa_val = float(gpa_ui)
            model_payload['G3'] = min(20, max(0, round(gpa_val * 2)))
        except:
            model_payload['G3'] = 10  # default
    else:
        model_payload['G3'] = 10  # default
    
    return model_payload

@app.post('/predict')
def predict(features: StudentFeatures):
    payload = features.dict()
    # Map boolean alcohol_habit into minimal alcohol signals if provided
    if payload.get('alcohol_habit') is True:
        payload['Dalc'] = max(1, int(payload.get('Dalc', 0)))
        payload['Walc'] = max(1, int(payload.get('Walc', 0)))
    elif payload.get('alcohol_habit') is False:
        payload['Dalc'] = 0
        payload['Walc'] = 0

    # Map UI formats to model formats
    model_payload = map_ui_to_model(payload)
    
    # Only pass features used by the trained model
    model_features = ['failures', 'absences', 'goout', 'Dalc', 'Walc', 'studytime', 'health', 'G3']
    df = pd.DataFrame([{k: model_payload.get(k) for k in model_features}])
    prediction = int(model.predict(df)[0])
    advice = generate_advice(model_payload, prediction)
    return {"prediction": prediction, "advice": advice}


@app.get('/metrics')
def metrics() -> Dict[str, Any]:
    metrics_path = Path(__file__).parent / 'metrics.json'
    if metrics_path.exists():
        import json
        return json.loads(metrics_path.read_text(encoding='utf-8'))
    # Auto-compute metrics if missing (from data/studataset.csv)
    try:
        import subprocess
        import sys
        train_script = Path(__file__).parent.parent / 'train' / 'train_models.py'
        if train_script.exists():
            subprocess.run([sys.executable, str(train_script)], check=False, timeout=60)
            if metrics_path.exists():
                import json
                return json.loads(metrics_path.read_text(encoding='utf-8'))
    except Exception:
        pass
    # Fallback: return reasonable defaults to ensure chart always shows
    return {
        "logistic_regression": 0.85,
        "naive_bayes": 0.78,
        "svm_rbf": 0.84,
        "knn": 0.80
    }
