# Smart Customer Support Agent - RL Environment

This repository contains our submission for **Round 1** of the Meta PyTorch Hackathon. 

## Project Overview

We built a custom Reinforcement Learning (RL) environment using the PyTorch ecosystem, `gymnasium`, and the `OpenEnv` framework. The environment simulates a customer support scenario where an AI agent must balance resolution speed, customer satisfaction, and escalation costs.

### Environment Dynamics
The environment simulates a customer support scenario over a maximum of 10 turns.
- **State Space:** `(issue_type, issue_complexity, customer_frustration, turns_elapsed)`
  - All values are normalized to `0.0 - 1.0`.
  - `issue_type`: 0.0 (Routine), 0.5 (Technical), or 1.0 (Billing).
- **Action Space:** 
  - `0`: Respond automatically (solves low complexity issues, otherwise increases frustration).
  - `1`: Ask for more info (lowers complexity safely, unless it's a Billing issue).
  - `2`: Escalate to human (100% resolution, but you only get points if escalating was necessary).
- **Tasks (Grader Setup):**
  - **Task 1 (Easy)**: Routine customer issue.
  - **Task 2 (Medium)**: Technical issue requiring clarification steps.
  - **Task 3 (Hard)**: Angry customer with a billing issue.
- **Objective:** The episode terminates strictly with a success score between `0.0` and `1.0`.

## Baseline Agent

We provide `inference.py` as a reference baseline agent using an OpenAI-compatible LLM:
```bash
export API_BASE_URL="http://localhost:8000"
export HF_TOKEN="your_hf_token"
export MODEL_NAME="gpt-4o-mini"
python inference.py
```

- [x] **OpenEnv Framework:** Environment is wrapped defensively in a FastAPI interface using `openenv-core`.
- [x] **Public GitHub Repo:** This repository contains the full source code.
- [x] **Requirements:** `requirements.txt` is provided in the root directory.
- [x] **Demo Script:** `demo.py` trains a PPO model and contrasts it to a baseline strategy.
- [x] **Hugging Face Spaces Deployment:** Prepared for deployment (see below).

## How to Run Locally

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Demo Script:**
   This will train the PPO Agent (`stable-baselines3`) and evaluate its metrics compared to a static baseline logic.
   ```bash
   python demo.py
   ```
3. **Start the OpenEnv HTTP Server:**
   ```bash
   python app.py
   ```
   Navigate to `http://localhost:8000` to interact with the environment via API.

3. Push this directory's files.
4. The Space will automatically run `uvicorn app:app --host 0.0.0.0 --port 7860` (ensure `app.py` is configured for the Hugging Face default port).
