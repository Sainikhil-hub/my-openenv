from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/reset")
def reset():
    return {"observation": [0.0, 0.0, 0.0, 0.0]}

@app.post("/step")
def step(action: dict):
    return {"observation": [0.0, 0.0, 0.0, 0.0], "reward": 0.0, "done": False}

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
