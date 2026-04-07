import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CustomerSupportEnv(gym.Env):
    """
    RL Environment for a Smart Customer Support Agent.
    
    Goal: Handle three types of tasks successfully using different strategies.
    
    Tasks:
    0 (Easy): Routine issue. Can be resolved automatically if attempted.
    1 (Medium): Technical issue. Needs clarification first, then can be resolved automatically.
    2 (Hard): Frustrated customer with billing issue. Must be escalated to avoid anger.
    """
    
    def __init__(self, max_turns=10):
        super(CustomerSupportEnv, self).__init__()
        
        self.max_turns = max_turns
        
        # State:
        # [issue_type (0=Routine, 0.5=Technical, 1.0=Billing),
        #  issue_complexity (float 0->1), 
        #  customer_frustration (float 0->1), 
        #  turns_elapsed_normalized (float 0->1)]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32), 
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32), 
            dtype=np.float32
        )
        
        # Actions:
        # 0: Respond Automatically (attempt to solve)
        # 1: Ask for More Info (clarify issue)
        # 2: Escalate to Human (defer to human support)
        self.action_space = spaces.Discrete(3)
        
        self.task_id = 0
        self.issue_type = 0.0
        self.issue_complexity = 0.0
        self.customer_frustration = 0.0
        self.turns_elapsed = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if options is None:
            options = {}
            
        # Default to a random task if not provided in options
        self.task_id = options.get('task_id', self.np_random.choice([0, 1, 2]))
        
        if self.task_id == 0:
            # Task 0 (Easy): Routine Issue
            self.issue_type = 0.0
            self.issue_complexity = 0.2
            self.customer_frustration = 0.1
        elif self.task_id == 1:
            # Task 1 (Medium): Technical Issue
            self.issue_type = 0.5
            self.issue_complexity = 0.8
            self.customer_frustration = 0.2
        else:
            # Task 2 (Hard): Billing Issue
            self.issue_type = 1.0
            self.issue_complexity = 0.9
            self.customer_frustration = 0.8
            
        self.turns_elapsed = 0
        
        return self._get_obs(), {}
        
    def _get_obs(self):
        return np.array([
            self.issue_type,
            self.issue_complexity, 
            self.customer_frustration, 
            self.turns_elapsed / self.max_turns
        ], dtype=np.float32)

    def step(self, action):
        self.turns_elapsed += 1
        
        reward = 0.0
        terminated = False
        truncated = False
        info = {'resolved': False, 'escalated': False, 'angry': False}
        
        if action == 0:
            # Respond Automatically
            if self.issue_complexity <= 0.3:
                # Issue simple enough to resolve automatically
                reward = 1.0  # Perfect score!
                terminated = True
                info['resolved'] = True
            else:
                # Failed attempt increases frustration
                self.customer_frustration = min(1.0, self.customer_frustration + 0.3)
                
        elif action == 1:
            # Ask for More Info
            if self.issue_type == 1.0: # Billing issue, asking info just angers them greatly
                self.customer_frustration = min(1.0, self.customer_frustration + 0.5)
            else:
                # Safely drops complexity, slightly increases frustration
                self.issue_complexity = max(0.1, self.issue_complexity - 0.3)
                self.customer_frustration = min(1.0, self.customer_frustration + 0.1)
                
        elif action == 2:
            # Escalate
            terminated = True
            info['escalated'] = True
            if self.issue_type == 1.0:
                reward = 1.0  # Escalating the hard billing issue is the correct move
            elif self.issue_type == 0.5:
                reward = 0.5  # Partial credit for escalating medium, but better to solve
            else:
                reward = 0.0  # Escalating easy issue gives 0

        # Check turn limit
        if not terminated and self.turns_elapsed >= self.max_turns:
            truncated = True
            reward = 0.0
            info['angry'] = True

        # Check frustration limit
        if self.customer_frustration >= 1.0 and not terminated:
            truncated = True
            reward = 0.0
            info['angry'] = True

        return self._get_obs(), float(reward), terminated, truncated, info
