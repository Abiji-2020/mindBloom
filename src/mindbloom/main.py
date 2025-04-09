import os 
from supabase import create_client, Client
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from mindbloom.stablity import emotion_stablity

# Load environment variables
load_dotenv()
url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# Example Supabase call (ensure this is not called on import in production)
response = supabase.table("children").select("*").execute()
print(response.data)

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
class EmotionalState(BaseModel):
    emotion: List[Dict[str, float]]  # e.g., [{"happy": 0.9}, {"sad": 0.1}]
    pause_frequency: float
    hand_movement_range: float
    hand_movement_speed: float
    completion_time: float

    hard_level: int
    level_abandonment_rate: float
    retry_attempts: Optional[int] = None
    # reaction_time = hand_movement_speed / hand_movement_range or completion_time 

# Emotion State Endpoint
@app.post("/emotion_state")
async def emotion_state(payload: EmotionalState):
    return {"received": payload.dict()}
