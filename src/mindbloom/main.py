import os 
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
from mindbloom.stablity import emotion_stablity
from mindbloom.focus import get_focus
from mindbloom.motor_engagement import get_mortor_engagement


sample_data = [
    [0.70, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    [0.60, 0.10, 0.05, 0.10, 0.05, 0.05, 0.05],
    [0.40, 0.15, 0.05, 0.15, 0.05, 0.05, 0.05],
    [0.40, 0.20, 0.05, 0.20, 0.05, 0.05, 0.05],
    [0.45, 0.25, 0.05, 0.15, 0.05, 0.05, 0.00],
    [0.30, 0.10, 0.05, 0.10, 0.10, 0.05, 0.05],
    [0.65, 0.05, 0.05, 0.05, 0.10, 0.05, 0.05],
    [0.60, 0.05, 0.10, 0.05, 0.05, 0.10, 0.05],
    [0.40, 0.10, 0.20, 0.05, 0.10, 0.10, 0.05],
    [0.20, 0.0, 0.25, 0.05, 0.00, 0.10, 0.05],
]

print(emotion_stablity(sample_data))
def min_max_normalize(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized = (arr - min_val) / (max_val - min_val + 1e-8)
    return normalized


# normalized
speed = [0.5, 0.6, 0.7, 0.8, 0.9]
range_ = [0.5, 0.6, 0.7, 0.8, 0.9]
symmetrry = [0.5, 0.6, 0.7, 0.8, 0.9]
# Get reaction time data
print("Normalize")
print(min_max_normalize(speed))

print(get_focus(sample_data, speed, range_, symmetrry))
print(get_mortor_engagement(speed, range_, symmetrry))
# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health Check Endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Define payload structure
class EmotionInput(BaseModel):
    emotion: List[List[float]]
    speed: List[float]
    ranges: List[float]
    symmetry: List[float]

# Emotion State Endpoint
@app.post("/emotion_state")
async def emotion_state(payload: EmotionInput):
    # Extract data from payload
    emotion_data = payload.emotion
    speed_data = payload.speed
    range_data = payload.ranges
    symmetry_data = payload.symmetry

    # Process data using the imported functions
    focus_score = get_focus(emotion_data, speed_data, range_data, symmetry_data)
    motor_engagement_score = get_mortor_engagement(speed_data, range_data, symmetry_data)
    emotion_stability_score = emotion_stablity(emotion_data)

    return {
        "focus_score": focus_score,
        "motor_engagement_score": motor_engagement_score,
        "emotion_stability_score": emotion_stability_score,
    }
