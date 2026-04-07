from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from src.environment import CustomerSupportEnv
import uvicorn

app = FastAPI(title="Smart Customer Support Agent Env")

# Initialize environment
env = CustomerSupportEnv()

class ActionModel(BaseModel):
    action: int

class ResetRequest(BaseModel):
    task_id: Optional[int] = None

class ResetResponse(BaseModel):
    observation: List[float]
    info: Dict[str, Any]

class StepResponse(BaseModel):
    observation: List[float]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]

class StateResponse(BaseModel):
    observation: List[float]

@app.get("/")
def read_root():
    return {"message": "OpenEnv HTTP interface for Customer Support Environment running"}

@app.post("/reset", response_model=ResetResponse)
def reset_env(body: Optional[ResetRequest] = None):
    options = {}
    if body and body.task_id is not None:
        if body.task_id not in [0, 1, 2]:
            raise HTTPException(status_code=400, detail="Invalid task_id. Must be 0, 1, or 2.")
        options['task_id'] = body.task_id
    obs, info = env.reset(options=options)
    return {"observation": obs.tolist(), "info": info}

@app.get("/state", response_model=StateResponse)
def get_state():
    obs = env._get_obs()
    return {"observation": obs.tolist()}

@app.post("/step", response_model=StepResponse)
def step_env(body: ActionModel):
    if body.action not in [0, 1, 2]:
        raise HTTPException(status_code=400, detail="Invalid action. Must be 0, 1, or 2.")
    
    obs, reward, terminated, truncated, info = env.step(body.action)
    
    return {
        "observation": obs.tolist(),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": info
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
