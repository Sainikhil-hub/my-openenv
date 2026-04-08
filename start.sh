#!/bin/bash
# Start FastAPI backend on port 8000 in background
uvicorn app:app --host 0.0.0.0 --port 8000 &

# Wait for the backend to be ready
sleep 3

# Start Gradio UI on port 7860 (HF Spaces required port)
python app_ui.py
