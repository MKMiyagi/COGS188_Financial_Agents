import gymnasium as gym
import numpy as np
import yfinance as yf
from gymnasium import spaces

class StockTrainingEnv(gym.Env):
    def __init__(self, tickers=[], start='2014-01-01', end='2024-01-01', initial_balance=10000, window_size=30):
        super(StockTrainingEnv, self).__init__()

        # Assign key variables
        self.tickers = tickers
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.possible_trades = [
            -1, -5, -10, -25, -50, -100, # Sell
            0, # Hold
            1, 5, 10, 25, 50, 100 # Buy
        ]

        # Download stock data from yfinance
        stock_data = yf.download(tickers, start=start, end=end, auto_adjust=True, actions=True)
        self.df = stock_data['Close'].dropna()
        self.num_stocks = len(tickers)

        # Store stock splits
        self.splits = {stock: stock_data['Stock Splits'][stock].dropna() for stock in tickers}

        # 3 Discrete actions, 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.MultiDiscrete([len(self.possible_trades)] * self.num_stocks)

        # Observation space is the stock prices from the past 30 days and the current balance
        self.observation_space = spaces.Box(low =-np.inf, high=np.inf, shape=([1, self.window_size * self.num_stocks + 1]), dtype=np.float32)

        self.profits = []
        self.trade_history = {stock: {"avg_price": 0.0, "shares": 0} for stock in self.tickers}

        # Ensure all variables are properly assigned
        self.reset()

    def reset(self, seed=None, options=None):
        # Assign key variables
        self.balance = self.initial_balance
        self.shares_held = {stock: 0 for stock in self.tickers}
        self.current_step = self.window_size
        self.done = False

        # Initialize trade history and profits
        self.trade_history = {stock: {"avg_price": 0.0, "shares": 0} for stock in self.tickers}
        self.profits = []

        return self._get_observation()
    
    def stock_split(self):
        for stock in self.tickers:
            if self.current_step in self.splits[stock].index:
                split_ratio = self.splits[stock].loc[self.current_step]
                if split_ratio > 0: 
                    self.shares_held[stock] *= split_ratio
    
    '''
    Take an action on each stock in tickers
    0 = Hold, 1 = Buy, 2 = Sell

    Parameters:
    action (list): List of actions to take on each stock in tickers
                    Formatted [action, num_shares, action, num_shares, ...]
    
    Returns:
    observation (np.array): The observation of the environment after taking the action
    '''
    def step(self, action):
        # Check if the episode has ended
        if self.done:
            return self._get_observation(), 0, True, False
        
        # Adjust for Stock Splits
        self.stock_split()

        # Find current prices of stocks
        current_prices = self.df.iloc[self.current_step]
        reward = 0  

        # Execute actions for each stock
        for i, ticker in enumerate(self.tickers):
            # Determine the number of shares to buy or sell
            num_shares = action[i]

            # Buy
            if num_shares > 0:
                # Maxmimum amount of stock we can purchase
                max_afford = self.balance // current_prices[ticker]

                # Limit the amount of shares we can buy
                shares_to_buy = min(max_afford, num_shares)

                if shares_to_buy < num_shares:
                    reward -= 3
                if shares_to_buy > 0:
                    self.shares_held[ticker] += shares_to_buy
                    self.balance -= shares_to_buy * current_prices[ticker]

                    # Update weighted average purchase price
                    prev_shares = self.trade_history[ticker]["shares"]
                    prev_avg = self.trade_history[ticker]["avg_price"]
                    new_total_shares = prev_shares + shares_to_buy
                    new_avg = (prev_shares * prev_avg + shares_to_buy * current_prices[ticker]) / new_total_shares if new_total_shares > 0 else current_prices[ticker]
                    self.trade_history[ticker]["shares"] = new_total_shares
                    self.trade_history[ticker]["avg_price"] = new_avg

                    reward += 5
                else:
                    reward -= 5
            
            # Sell
            elif num_shares < 0 and self.shares_held[ticker] > 0:
                shares_to_sell = min(self.shares_held[ticker], abs(num_shares))
                if shares_to_sell > 0:
                    self.shares_held[ticker] -= shares_to_sell
                    self.balance += shares_to_sell * current_prices[ticker]
                    
                    # Calculate profit using the average cost basis
                    avg_buy_price = self.trade_history[ticker]["avg_price"]
                    profit = shares_to_sell * (current_prices[ticker] - avg_buy_price)
                    self.profits.append(profit)
                    self.trade_history[ticker]["shares"] -= shares_to_sell
                    if self.trade_history[ticker]["shares"] == 0:
                        self.trade_history[ticker]["avg_price"] = 0.0

                    if profit > 0:
                        reward += 5
                    else:
                        reward -= 2
                else:
                    reward -= 5
            
        # Move to next step
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self.done = True
        
        # Calculate reward based on portfolio value change
        total_value = self.balance + sum(self.shares_held[stock] * current_prices[stock] for stock in self.tickers)
        reward += (total_value - self.initial_balance) / self.initial_balance * 10
        return self._get_observation(), reward, self.done, False

    '''
    Return the observation of the environment
    30 days of stock prices for each stock in tickers

    Returns:
    observation (np.array): The observation of the environment
    '''
    def _get_observation(self):
        # Get the past 30 days of prices for all stocks
        obs = self.df.iloc[self.current_step - self.window_size:self.current_step].values.flatten()
        obs = np.append(obs, [self.balance])
        return np.array(obs, dtype=np.float32)
    
    def render(self):
        # Visualize env state at each step
        total_value = self.balance + sum(self.shares_held[t] * self.df.iloc[self.current_step][t] for t in self.tickers)
        print(f"Step: {self.current_step}, Balance: {self.balance}, Total Value: {total_value}, Shares: {self.shares_held}")