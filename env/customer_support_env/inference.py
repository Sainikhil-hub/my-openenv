import os
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

client = OpenAI(api_key=OPENAI_API_KEY)

def get_action(obs):
    # Determine the action using the LLM given the state

    prompt = f"""
    The customer support environment state is: {obs}.
    The state structure is [issue_type, issue_complexity, customer_frustration, turns_elapsed].
    All values are normalized between 0.0 and 1.0.

    Issue Type (0.0=Routine, 0.5=Technical, 1.0=Billing)
    
    Actions:
    0: Respond Automatically. Attempt to resolve. Do this if complexity is low (<= 0.3). Otherwise increases frustration.
    1: Ask for More Info. Lowers complexity, increases frustration a bit. Do NOT use this for Billing issues!
    2: Escalate to Human. Always solvs the problem. High score for escalating Billing, partial for Technical, 0 for Routine.

    Which action (0, 1, or 2) should the agent take? Reply ONLY with the integer.
    """
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    
    try:
        action = int(response.choices[0].message.content.strip())
        if action not in [0, 1, 2]:
            action = 0
    except ValueError:
        action = 0
        
    return action

def main():
    print("[START]")
    
    # Initialize the environment
    res = requests.post(f"{API_BASE_URL}/reset")
    if res.status_code != 200:
        print(f"Failed to reset environment: {res.text}")
        return
        
    data = res.json()
    obs = data["observation"]
    
    terminated = False
    truncated = False
    
    while not terminated and not truncated:
        action = get_action(obs)
        res = requests.post(f"{API_BASE_URL}/step", json={"action": action})
        if res.status_code != 200:
            print(f"Failed to step environment: {res.text}")
            break
            
        data = res.json()
        reward = data["reward"]
        terminated = data["terminated"]
        truncated = data["truncated"]
        obs = data["observation"]
        
        print(f"[STEP] Action: {action} | Reward: {reward} | Terminated: {terminated} | Truncated: {truncated} | Obs: {obs}")
        
    print(f"[END] Total Reward: {reward}")

if __name__ == "__main__":
    main()
