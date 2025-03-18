import numpy as np
import random
from tqdm import tqdm

# Define Actions: Hold (0), Buy (1), Sell (2)
ACTIONS = [0, 1, 2]

# Monte Carlo Training Function
def monte_carlo_train(env, episodes=1000, gamma=0.95, lr=0.1):
    """ Trains a trading agent using Monte Carlo RL inside a Gym environment """
    Q = {}  # Initialize Q-table

    for _ in tqdm(range(episodes), desc="Monte Carlo Training", dynamic_ncols=True):
        state = env.reset()  # Reset environment for new episode
        states, actions, rewards = [], [], []
        done = False

        while not done:
            action = env.action_space.sample() # Random action (exploration)
            for i in range(len(action)):
                action[i] = env.possible_trades[action[i]]
            next_state, reward, done, _ = env.step(action)

            # Store episode history
            states.append(tuple(state))
            actions.append(tuple(action))
            rewards.append(reward)
            state = next_state  # Move to next state

        # Monte Carlo update: Compute returns and update Q-values
        G = 0  # Return value
        memory = set()
        for t in reversed(range(len(states))):
            G = gamma * G + rewards[t]
            if (states[t], actions[t]) not in memory:
                memory.add((states[t], actions[t]))
                if (states[t]) not in Q.keys():
                    Q[states[t]] = {}
                if (actions[t]) not in Q[states[t]].keys():
                    Q[states[t]][actions[t]] = 0

                Q[states[t]][actions[t]] = Q[states[t]][actions[t]] + lr * (G - Q[states[t]][actions[t]])

    print("Monte Carlo training complete")
    return Q  # Return trained Q-table
# Evaluate the trained model
def monte_carlo_eval(env, Q, model_name="Monte Carlo Agent"):
    """ Runs the trained agent on the Gym environment to evaluate performance """
    state = env.reset()
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

        next_state, reward, done, _ = env.step(action)
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