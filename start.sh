#!/bin/bash
# Single process: FastAPI + Gradio UI both served by uvicorn on port 7860
uvicorn app:app --host 0.0.0.0 --port 7860
