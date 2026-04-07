# 🎧 Smart Customer Support Agent - RL Environment

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains our custom Reinforcement Learning (RL) and Generative AI environment submission for **Round 1** of the Meta PyTorch Hackathon. 

## 📖 Project Overview

We built a custom `gymnasium` environment wrapped inside an `OpenEnv` compliant HTTP interface using FastAPI and PyTorch. The environment simulates a customer support scenario where an AI agent must balance resolution speed, customer satisfaction, and escalation protocols.

### Environment Dynamics
The environment simulates a customer support session with a hard limit of 10 conversational turns.
- **State Space (Box):** `[issue_type, issue_complexity, customer_frustration, turns_elapsed]`
  - All output arrays are strictly normalized between `0.0` and `1.0`.
  - `issue_type`: Category indicator (Routine = 0.0, Technical = 0.5, Billing = 1.0).
- **Action Space (Discrete):**
  - `0 (Respond Automatically)`: Explains technical details. Attempts to resolve low-complexity issues directly, but angers the customer if the complexity is still too high.
  - `1 (Ask for More Info)`: Lowers the complexity of a problem safely. *(Warning: Doing this on an angry billing customer will maximize frustration!)*
  - `2 (Escalate to Human)`: Immediately resolves the incident, but provides limited points because human capital is expensive.

### Hackathon Grader Evaluation (3 Tasks)
To comply with the Round 1 Meta rules, the environment emits a final evaluation score strictly bounded between `0.0` and `1.0` upon successful termination across three specific scenarios:
- **Task 0 (Easy)**: Routine customer issue.
- **Task 1 (Medium)**: Technical issue requiring step-by-step clarification.
- **Task 2 (Hard)**: Angry customer experiencing a billing issue requiring immediate human escalation.

---

## 🚀 Getting Started

Ensure you have Python 3.10+ installed.

### 1. Install Dependencies
```bash
git clone https://github.com/Sainikhil-hub/Create_environment.git
cd Create_environment
pip install -r requirements.txt
