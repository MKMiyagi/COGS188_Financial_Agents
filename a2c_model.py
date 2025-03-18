import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import your trading environment
from training_env import StockTrainingEnv  # Adjust the import based on your actual environment class name

# Set up device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ActorCriticNetwork(nn.Module):
    """
    Combined actor-critic network for A2C algorithm
    """
    def __init__(self, input_dim, n_actions, hidden_dim=128):
        super(ActorCriticNetwork, self).__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy head (actor)
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
        
        # Value head (critic)
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.to(device)
    
    def forward(self, x):
        features = self.shared(x)
        action_probs = F.softmax(self.policy(features), dim=-1)
        state_values = self.value(features)
        return action_probs, state_values

class A2CAgent:
    def __init__(self, env, learning_rate=1e-4, gamma=0.99, entropy_coef=0.01):
        self.env = env
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.device = device
        
        # Get observation space and action space dimensions
        if hasattr(env.observation_space, 'shape'):
            self.input_dim = np.prod(env.observation_space.shape)
        else:
            # Handle other observation space types as needed
            self.input_dim = env.observation_space.shape[0]
        
        # Store the number of stocks for action creation
        self.num_stocks = len(env.tickers)
        self.actions_per_stock = env.action_space.nvec[0]
        
        # Initialize the actor-critic network
        self.model = ActorCriticNetwork(self.input_dim, self.actions_per_stock).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def select_action(self, state, training=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_probs, _ = self.model(state_tensor)
        
        # Create a categorical distribution over the action probabilities
        dist = Categorical(action_probs)
        
        if training:
            # Sample an action during training
            action_idx = dist.sample()
        else:
            # Choose the most probable action during evaluation
            action_idx = torch.argmax(action_probs)
        
        # Convert the single action index to a vector of actions
        action = [action_idx.item()] * self.num_stocks
        
        return action, dist.log_prob(action_idx), dist.entropy()
    
def a2c_train(env, agent, n_episodes=1000, max_steps=200):
    episode_rewards = []
    
    # Create a progress bar with iteration speed
    pbar = tqdm(
        range(n_episodes), 
        desc="A2C Training",
    )
    
    for episode in pbar:
        state = env.reset()
        done = False
        episode_reward = 0
        
        # Lists to store episode data
        log_probs = []
        values = []
        rewards = []
        entropies = []
        
        step = 0
        while not done and step < max_steps:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # Get action probabilities and state value
            action_probs, state_value = agent.model(state_tensor)
            
            # Sample action
            dist = Categorical(action_probs)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
            entropy = dist.entropy()
            
            # Convert to action vector for the environment
            action = [action_idx.item()] * agent.num_stocks
            
            # Take action in the environment
            next_state, reward, terminated, truncated = env.step(action)
            done = terminated or truncated
            
            # Store step data
            log_probs.append(log_prob)
            values.append(state_value)
            rewards.append(reward)
            entropies.append(entropy)
            
            # Update state and counters
            state = next_state
            episode_reward += reward
            step += 1
        
        # Convert lists to tensors first
        log_probs = torch.cat(log_probs)
        values = torch.cat(values).squeeze()
        entropies = torch.stack(entropies)
        
        # Calculate returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + agent.gamma * R
            returns.insert(0, R)
        
        # Normalize returns
        returns = torch.FloatTensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate advantages using tensors
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Calculate losses
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = F.mse_loss(values, returns)
        entropy_loss = -entropies.mean()
        
        # Total loss
        loss = actor_loss + 0.5 * critic_loss + agent.entropy_coef * entropy_loss
        
        # Optimize
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()
        
        episode_rewards.append(episode_reward)
        avg_reward = episode_reward/step
        
        # Update progress bar with current reward and steps
        pbar.set_postfix({
            'reward': f'{avg_reward:.2f}',
            'steps': step
        })
    
    # Save the final model
    torch.save(agent.model.state_dict(), "a2c_model_final.pth")
    return episode_rewards

def a2c_eval(env, agent, n_episodes=10):
    """Evaluate the trained agent"""
    total_rewards = []
    
    # Create a progress bar for evaluation with iteration speed
    pbar = tqdm(
        range(n_episodes), 
        desc="Evaluating A2C Agent",
    )
    
    for _ in pbar:
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Select action without exploration
            action, _, _ = agent.select_action(state, training=False)
            
            # Take action in the environment
            next_state, reward, terminated, truncated = env.step(action)
            done = terminated or truncated
            
            # Update state and reward
            state = next_state
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        
        # Update progress bar
        pbar.set_postfix({'reward': f'{episode_reward:.2f}'})
    env.render()