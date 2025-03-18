import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from tqdm import tqdm
from training_env import StockTrainingEnv

# Define Policy Network (Neural Network for Action Selection)
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, num_tickers, num_actions, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Create separate output layers for each ticker
        self.action_heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_actions) for _ in range(num_tickers)
        ])
        
        self.softmax = nn.Softmax(dim=-1)
        self.num_tickers = num_tickers
        self.num_actions = num_actions

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        
        # Get action probabilities for each ticker
        action_probs = []
        for i in range(self.num_tickers):
            action_probs.append(self.softmax(self.action_heads[i](x)))
            
        return action_probs  # List of probability distributions for each ticker

# Policy Gradient Algorithm (REINFORCE)
class PolicyGradientAgent:
    def __init__(self, state_dim, num_tickers, possible_trades, lr=0.001, gamma=0.99):
        self.gamma = gamma
        self.memory = deque(maxlen=5000)  # Store trajectories
        self.num_tickers = num_tickers
        self.possible_trades = possible_trades
        num_actions = len(self.possible_trades)  # Fixed: each ticker has its own probability distribution over actions
        self.policy_net = PolicyNetwork(state_dim, num_tickers, num_actions)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action_probs_list = self.policy_net(state)
        
        actions = []
        action_probs = []
        
        # For each ticker, sample an action from its probability distribution
        for ticker_probs in action_probs_list:
            probs = ticker_probs.detach().numpy()
            action_idx = np.random.choice(len(probs), p=probs)
            
            # Convert action index to actual trade value from possible_trades
            action_value = self.possible_trades[action_idx]
            actions.append(action_value)
            action_probs.append(ticker_probs[action_idx])
            
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
            episode_loss = sum([-torch.log(prob) * G for prob in action_probs])
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

def train_policy_gradient(env, agent, episodes=1000, gamma=0.99, lr=0.001):
    for episode in tqdm(range(episodes), desc="Training Policy Gradient"):
        state = env.reset()
        done = False
        episode_reward = 0
        num_steps = 0
        
        while not done:
            actions, action_probs = agent.select_action(state)
            next_state, reward, done, _ = env.step(actions)
            agent.store_transition((state, action_probs, reward))
            state = next_state
            episode_reward += reward
            num_steps += 1
        agent.update_policy()
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {episode_reward}, Avg Reward: {episode_reward / num_steps}")

    agent.save_model("policy_gradient_model.pth")

# Initialize environment and agent
env = StockTrainingEnv(tickers=["AAPL", "TSLA", "META"])

# Define dimensions for the agent
state_dim = env.observation_space.shape[1]
num_tickers = len(env.tickers)
possible_trades = env.possible_trades

# Create agent
agent = PolicyGradientAgent(
    state_dim=state_dim,
    num_tickers=num_tickers,
    possible_trades=possible_trades
)

# Train agent
train_policy_gradient(env, agent, episodes=1000, gamma=0.99, lr=0.001)