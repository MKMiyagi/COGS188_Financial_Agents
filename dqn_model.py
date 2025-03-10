import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

class Stocktrader(nn.Module):
    def __init__(self, input_dim, output_dim, lr=0.001, step_size = 1, gamma=0.99):
        super(Stocktrader, self).__init__()
        self.hidden1 = nn.Linear(input_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.hidden3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, output_dim)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.out(x)
        return x

def train(model, target_model, train_env,
          num_episodes=500, batch_size=32, gamma=0.99,
          epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995, target_update=10):
    
    optimizer = model.optimizer
    memory = deque(maxlen=10000)
    epsilon = epsilon_start
    
    for episode in range(num_episodes):
        state = train_env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        total_reward = 0

        while True:
            # e-greedy policy
            if random.random() < epsilon:
                action = train_env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = model(state)
                    action = torch.argmax(q_values).item()
            
            next_state, reward, done, _ = train_env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            memory.append((state, action, reward, next_state, done))
            
            state = next_state
            total_reward += reward
            
            if done:
                break

        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.cat(states)
            actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
            next_states = torch.cat(next_states)
            dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
            
            # Calculate Q-values
            q_values = model(states).gather(1, actions)
            next_q_values = target_model(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + (gamma * next_q_values * (1 - dones))
            
            # Calculate loss and update policy
            loss = F.mse_loss(q_values, target_q_values.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if episode % target_update == 0:
            target_model.load_state_dict(model.state_dict())
        
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        model.scheduler.step()
        
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")
