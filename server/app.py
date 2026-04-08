from fastapi import FastAPI
import uvicorn

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
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == '__main__':
    main()
