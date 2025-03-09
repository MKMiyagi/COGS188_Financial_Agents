import gymnasium as gym
import numpy as np
import yfinance as yf
from gymnasium import spaces

class StockTrainingEnv(gym.ENV):
    def __init__(self, tickers=[], start='2014-01-01', end='2024-01-01', initial_balance=10000, window_size=30, max_shares_per_trade = 5):
        super(StockTrainingEnv, self).__init__()

        # Assign key variables
        self.tickers = tickers
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.max_shares_per_trade = max_shares_per_trade

        # Download stock data from yfinance
        self.df = yf.download(tickers, start=start, end=end)['Close'].dropna()
        self.num_stocks = len(tickers)

        # Define an action and observation space
        # 3 Discrete actions, 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.MultiDiscrete([3, max_shares_per_trade + 1] * self.num_stocks)
        self.observation_space = spaces.Box(low =-np.inf, high=np.inf, shape=(self.window_size * num_stocks + 1), dtype=np.float32)

        # Ensure all variables are properly assigned
        self.reset()

    def reset(self, seed=None, options=None):
        # Assign key variables
        self.balance = self.initial_balance
        self.shares_held = {ticker: 0 for stock in self.ticker}
        self.current_step = self.window_size
        self.done = False

        return self._get_observation(), {}
    
    def step(self, action):
        # Check if the episode has ended
        if self.done:
            return self._get_observation(), 0, True, False, {}
        
        # Find current prices of stocks
        current_prices = self.df.iloc[self.current_step]

        # Execute actions for each stock
        for i, ticker in enumerate(self.tickers):
            # Determine action 0 = Hold, 1 = Buy, 2 = Sell
            action_type = action[i * 2]

            # Determine the number of shares to buy
            num_shares = action[i*2 + 1]

            # Buy
            if action_type == 2 and num_shares > 0:
                # Maxmimum amount of stock we can purchase
                max_afford = self.balance // current_prices[ticker]

                # Limit the amount of shares we can buy
                shares_to_buy = min(_max_afford, num_shares)

                if shares_to_buy > 0:
                    self.shares_held[ticker] += shares_to_buy
                    self.balance
        
        # Move to next step
        self.current_step += 1
        if self.current_step > len(self.df) - 1:
            self.done = True
        
        # Calculate reward based on portfolio value change
        total_value = self.balance + sum(self.shares_held[stock] * current_prices[stock] for stock in self.tickers)
        reward = total_value - self.initial_balance
        return self._get_observation(), reward, self.done, False, {}

    def _get_observation(self):
        # Get the past 30 days of prices for all stocks
        obs = self.df.iloc[self.current_step - self.window:self.current_step].values.flatten()
        obs = np.append(obs, [self.balance])
        return np.array(obs, dtype=np.float32)
    
    def render(self):
        # Visualize env state at each step
        total_value = self.balance + sum(self.shares_held[t] * self.df.iloc[self.current_step][t] for t in self.tickers)
        print(f"Step: {self.current_step}, Balance: {self.balance}, Total Value: {total_value}, Shares: {self.shares_held}")




    
        


