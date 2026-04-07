import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from src.environment import CustomerSupportEnv
import numpy as np
import matplotlib.pyplot as plt

def run_demo():
    print("="*50)
    print("Starting Smart Customer Support Agent Demo")
    print("="*50)

    # 1. Initialize custom environment
    env = CustomerSupportEnv()
    
    # 2. Verify Gymnasium compliance
    print("Checking environment compatibility...")
    check_env(env)
    print("Environment is valid!\n")
    
    # 3. Initialize RL Agent using PPO
    print("Training PPO Agent (this may take a few seconds)...")
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=15000)
    print("Training Complete!\n")
    
    # 4. Evaluate Agent against Rule-Based Baseline
    print("="*50)
    print("Evaluating Agent vs Baseline (100 episodes)")
    print("="*50)
    
    def evaluate(policy_fn, episodes=100):
        resolutions = 0
        escalations = 0
        total_time = 0
        total_rewards = 0
        episode_rewards = []
        
        for _ in range(episodes):
            obs, info = env.reset()
            done = False
            ep_reward = 0
            
            while not done:
                action = policy_fn(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                done = terminated or truncated
                
                if done:
                    total_time += env.turns_elapsed
                    if info.get('resolved', False):
                        resolutions += 1
                    if info.get('escalated', False):
                        escalations += 1
            
            episode_rewards.append(ep_reward)
            total_rewards += ep_reward
            
        avg_time = total_time / episodes
        resolution_rate = resolutions / episodes
        escalation_rate = escalations / episodes
        avg_reward = total_rewards / episodes
        
        return resolution_rate, escalation_rate, avg_time, avg_reward, episode_rewards

    # Baseline Policy: Always try to Auto-Respond once, then Escalate
    def baseline_policy(obs):
        # obs is now [type, complexity, frustration, turns]
        turns_elapsed_norm = obs[3]
        if turns_elapsed_norm == 0.0:
            return 0  # Auto-Respond first
        return 2      # Escalate otherwise
        
    # RL Agent Policy
    def agent_policy(obs):
        action, _states = model.predict(obs, deterministic=True)
        return int(action)

    # Run evaluations
    base_res, base_esc, base_time, base_reward, base_rewards_list = evaluate(baseline_policy)
    agent_res, agent_esc, agent_time, agent_reward, agent_rewards_list = evaluate(agent_policy)
    
    print(f"{'Metric':<20} | {'Rule-Based Baseline':<20} | {'RL Agent (PPO)':<20}")
    print("-" * 65)
    print(f"{'Resolution Rate':<20} | {base_res*100:>19.1f}% | {agent_res*100:>19.1f}%")
    print(f"{'Escalation Rate':<20} | {base_esc*100:>19.1f}% | {agent_esc*100:>19.1f}%")
    print(f"{'Avg Response Time':<20} | {base_time:>19.2f}T | {agent_time:>19.2f}T")
    print(f"{'Avg Episode Reward':<20} | {base_reward:>19.2f} | {agent_reward:>19.2f}")
    
    # Generate Matplotlib chart
    plt.figure(figsize=(10, 5))
    window = 10
    
    # Calculate moving averages for smoother lines
    def moving_avg(data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
        
    base_smooth = moving_avg(base_rewards_list, window)
    agent_smooth = moving_avg(agent_rewards_list, window)
    
    plt.plot(agent_smooth, label='RL Agent (PPO)', color='blue')
    plt.plot(base_smooth, label='Baseline', color='red', linestyle='--')
    plt.title('Agent Performance vs Baseline (100 Eval Episodes Moving Avg)')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_results.png')
    print("\nVisual results saved to 'training_results.png'!")
    print("Demo completed successfully!")

if __name__ == "__main__":
    run_demo()
