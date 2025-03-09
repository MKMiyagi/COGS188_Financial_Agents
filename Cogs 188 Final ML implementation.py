import numpy as np
import random

# Define Actions: Buy, Sell, Hold
ACTIONS = ["Buy", "Sell", "Hold"]

# Monte Carlo Training Function
def monte_carlo_train(env, episodes=1000, gamma=0.95):
    """ Trains a trading agent using Monte Carlo RL inside a Gym environment """
    Q = {}  # Initialize Q-table

    for episode in range(episodes):
        state, _ = env.reset()  # Reset environment for new episode
        states, actions, rewards = [], [], []
        done = False

        while not done:
            action = env.action_space.sample()  # Random action (exploration)
            next_state, reward, done, _, _ = env.step(action)

            # Store episode history
            states.append(tuple(state))
            actions.append(tuple(action))
            rewards.append(reward)
            state = next_state  # Move to next state

        # Monte Carlo update: Compute returns and update Q-values
        G = 0  # Return value
        for t in reversed(range(len(states))):
            G = gamma * G + rewards[t]
            Q[(states[t], actions[t])] = Q.get((states[t], actions[t]), 0) + 0.1 * (G - Q.get((states[t], actions[t]), 0))

    print("✅ Monte Carlo training complete!")
    return Q  # Return trained Q-table

# Q-Learning Training Function
def q_learning_train(env, episodes=1000, alpha=0.1, gamma=0.95, epsilon=0.1):
    """ Trains a trading agent using Q-Learning inside a Gym environment """
    Q = {}  # Initialize Q-table

    for episode in range(episodes):
        state, _ = env.reset()
        done = False

        while not done:
            # ε-greedy policy: Explore or exploit
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = max(ACTIONS, key=lambda a: Q.get((tuple(state), a), 0))  # Exploit

            next_state, reward, done, _, _ = env.step(action)

            # Q-value update using Bellman Equation
            best_next_action = max(ACTIONS, key=lambda a: Q.get((tuple(next_state), a), 0))
            Q[(tuple(state), tuple(action))] = Q.get((tuple(state), tuple(action)), 0) + \
                alpha * (reward + gamma * Q.get((tuple(next_state), best_next_action), 0) - Q.get((tuple(state), tuple(action)), 0))

            state = next_state  # Move to next state

    print("✅ Q-Learning training complete!")
    return Q  # Return trained Q-table

# Evaluate the trained model
def evaluate_agent(env, Q, model_name="Agent"):
    """ Runs the trained agent on the Gym environment to evaluate performance """
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = max(ACTIONS, key=lambda a: Q.get((tuple(state), a), 0))
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        state = next_state

    print(f"✅ {model_name} Final Portfolio Value: {total_reward:.2f}")
    return total_reward


# once the Gym environmentis done, you can train and test the RL agent with the following
from stock_env import StockTrainingEnv 

# Create the trading environment
env = StockTrainingEnv(tickers=["AAPL", "TSLA"])

# Train Monte Carlo agent
Q_mc = monte_carlo_train(env, episodes=1000)

# Train Q-Learning agent
Q_ql = q_learning_train(env, episodes=1000)

# Evaluate Monte Carlo
evaluate_agent(env, Q_mc, "Monte Carlo Agent")

# Evaluate Q-Learning
evaluate_agent(env, Q_ql, "Q-Learning Agent")
