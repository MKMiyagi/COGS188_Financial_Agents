import numpy as np
import random
from tqdm import tqdm
import torch

# Define Actions: Hold (0), Buy (1), Sell (2)
ACTIONS = [0, 1, 2]

# Monte Carlo Training Function
def monte_carlo_train(env, episodes=1000, gamma=0.95, lr=0.1):
    """ Trains a trading agent using Monte Carlo RL inside a Gym environment """
    Q = {}  # Initialize Q-table

    for episode in tqdm(range(episodes), desc="Monte Carlo Training", dynamic_ncols=True):
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
            if (states[t]) not in Q.keys():
                Q[states[t]] = {}
            if (actions[t]) not in Q[states[t]].keys():
                Q[states[t]][actions[t]] = 0

            Q[states[t]][actions[t]] = Q[states[t]][actions[t]] + lr * (G - Q[states[t]][actions[t]])

    print("Monte Carlo training complete")
    return Q  # Return trained Q-table

# Q-Learning Training Function
def q_learning_train(env, episodes=1000, alpha=0.1, gamma=0.95, epsilon=0.1):
    """ Trains a trading agent using Q-Learning inside a Gym environment """
    Q = {}  # Initialize Q-table

    for episode in tqdm(range(episodes), desc="Monte Carlo Training", dynamic_ncols=True):
        state, _ = env.reset()
        done = False

        while not done:
            # ε-greedy policy: Explore or exploit
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                all_possible_actions = Q.get(tuple(state), {})
                if len(all_possible_actions) == 0:
                    # If the state is not in the Q-table, take a random action
                    action = env.action_space.sample()
                else:
                    # Select the action with the highest Q-value
                    action = max(all_possible_actions.keys(), key=lambda a: all_possible_actions[a])

            next_state, reward, done, _, _ = env.step(action)

            # Find best next action
            all_possible_next_actions = Q.get(tuple(next_state), {})
            if len(all_possible_next_actions) == 0:
                # If the state is not in the Q-table, take a random action
                best_next_action = env.action_space.sample()
            else:
                # Select the action with the highest Q-value
                best_next_action = max(all_possible_next_actions.keys(), key=lambda a: all_possible_next_actions[a])

            # Update Q-value
            state = tuple(state)
            next_state = tuple(next_state)
            action = tuple(action)
            best_next_action = tuple(best_next_action)
            if (state) not in Q.keys():
                Q[state] = {}
            if (next_state) not in Q.keys():
                Q[next_state] = {best_next_action: 0}
            Q[state][action] = Q[state][action] + alpha * (reward + gamma * Q[next_state][best_next_action] - Q[state][action])

            state = next_state  # Move to next state

    print("Q-Learning training complete")
    return Q  # Return trained Q-table

# Evaluate the trained model
def evaluate_agent(env, Q, model_name="Agent"):
    """ Runs the trained agent on the Gym environment to evaluate performance """
    state, _ = env.reset()
    done = False
    total_reward = 0
    step = 1

    while not done:
        # Get all actions seen during MC training
        all_possible_actions = Q.get(tuple(state), {})

        if len(all_possible_actions) == 0:
            # If the state is not in the Q-table, take a random action
            action = env.action_space.sample()
            print(f"Taking random action on step {step}")
        else:
            # Select the action with the highest Q-value
            action = max(all_possible_actions.keys(), key=lambda a: all_possible_actions[a])

        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        state = next_state

        # Display results every 20 steps
        if step % 20 == 0:
            env.render()

        step += 1

    print(f"{model_name} Results:")
    env.render()
    print(f"{model_name} Final Portfolio Value: {total_reward:.2f}")
    return total_reward


# once the Gym environmentis done, you can train and test the RL agent with the following
from training_env import StockTrainingEnv 

# Create the trading environment
env = StockTrainingEnv(tickers=["AAPL", "TSLA", "META"])

# Train Monte Carlo agent
Q_mc = monte_carlo_train(env, episodes=3000)

# Evaluate Monte Carlo
evaluate_agent(env, Q_mc, "Monte Carlo Agent")