import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
from tqdm import tqdm
from training_env import StockTrainingEnv

# Define Policy Network (Neural Network for Action Selection)
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, num_tickers, num_actions, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Create separate output layers for each ticker
        self.action_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, num_actions)
            ) for _ in range(num_tickers)
        ])
        
        self.num_tickers = num_tickers
        self.num_actions = num_actions

    def forward(self, state):
        x = self.network(state)
        
        # Get action probabilities for each ticker
        action_probs = []
        for i in range(self.num_tickers):
            action_probs.append(nn.functional.softmax(self.action_heads[i](x), dim=-1))
            
        return action_probs  # List of probability distributions for each ticker

# Policy Gradient Algorithm (REINFORCE)
class PolicyGradientAgent:
    def __init__(self, env, lr=0.001, gamma=0.99):
        self.env = env
        self.gamma = gamma
        self.memory = deque(maxlen=5000)  # Store trajectories
        self.possible_trades = env.possible_trades
        self.num_actions = env.action_space.nvec[0]
        self.num_tickers = len(env.tickers)

        # Get observation space and action space dimensions
        if hasattr(env.observation_space, 'shape'):
            self.input_dim = np.prod(env.observation_space.shape)
        else:
            # Handle other observation space types as needed
            self.input_dim = env.observation_space.shape[0]

        self.policy_net = PolicyNetwork(self.input_dim, self.num_tickers, self.num_actions)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def select_action(self, state, training=True):
        state = torch.tensor(state, dtype=torch.float32)
        action_probs_list = self.policy_net(state)
        
        actions = []
        action_probs = []
        
        # For each ticker, sample an action from its probability distribution
        for ticker_probs in action_probs_list:
            dist = Categorical(ticker_probs)
            if training:
                action = dist.sample()
            else:
                action = torch.argmax(ticker_probs)
            
            # Convert action index to actual trade value from possible_trades
            actions.append(self.env.possible_trades[action.item()])
            action_probs.append(dist.log_prob(action))
            
        return actions, action_probs

    def store_transition(self, transition):
        self.memory.append(transition)

    def update_policy(self):
        """ Updates policy using Monte Carlo returns (REINFORCE). """
        R = 0
        policy_loss = []
        returns = []
        
        # Compute discounted rewards
        for _, _, reward in reversed(self.memory):
            R = reward + self.gamma * R
            returns.insert(0, R)  # Store return at t

        # Normalize returns for stability
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        for (state, action_probs, _), G in zip(self.memory, returns):
            # Sum up log probabilities of each ticker's action
            episode_loss = sum([-prob * G for prob in action_probs])
            policy_loss.append(episode_loss)

        # Backpropagation
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        self.memory.clear()  # Clear memory after update

    def save_model(self, path="policy_gradient_model.pth"):
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path="policy_gradient_model.pth"):
        self.policy_net.load_state_dict(torch.load(path))

def policy_gradient_train(env, agent, episodes=1000, gamma=0.99, lr=0.001):
    # Create a progress bar with iteration speed
    pbar = tqdm(
        range(episodes), 
        desc="Training Policy Gradient",
    )
    
    for episode in pbar:
        state = env.reset()
        done = False
        episode_reward = 0
        num_steps = 0
        
        while not done:
            actions, action_probs = agent.select_action(state, training=True)
            next_state, reward, done, _ = env.step(actions)
            agent.store_transition((state, action_probs, reward))
            state = next_state
            episode_reward += reward
            num_steps += 1
        
        agent.update_policy()
        
        # Update progress bar with current reward and average reward
        avg_reward = episode_reward / num_steps if num_steps > 0 else 0
        pbar.set_postfix({
            'total_reward': f'{episode_reward:.2f}',
            'avg_reward': f'{avg_reward:.2f}'
        })

    agent.save_model("policy_gradient_model.pth")

def policy_gradient_eval(env, agent, n_episodes=10):
    """Evaluate the trained agent"""
    total_rewards = []
    
    # Create a progress bar for evaluation with iteration speed
    pbar = tqdm(
        range(n_episodes), 
        desc="Evaluating Policy Gradient Agent",
    )
    
    for _ in pbar:
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Select action without exploration
            action, _ = agent.select_action(state, training=False)
            
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

# # Initialize environment and agent
# env = StockTrainingEnv(tickers=["AAPL", "TSLA", "META"])

# # Define dimensions for the agent
# state_dim = env.observation_space.shape[1]
# num_tickers = len(env.tickers)
# possible_trades = env.possible_trades

# # Create agent
# agent = PolicyGradientAgent(
#     state_dim=state_dim,
#     num_tickers=num_tickers,
#     possible_trades=possible_trades
# )

# # Train agent
# policy_gradient_train(env, agent, episodes=100, gamma=0.99, lr=0.001)

# print(env.profits)

# policy_gradient_eval(env, agent)