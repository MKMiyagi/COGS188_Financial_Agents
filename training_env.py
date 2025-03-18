import gymnasium as gym
import numpy as np
import yfinance as yf
from gymnasium import spaces
import matplotlib.pyplot as plt
from IPython.display import clear_output

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

        # Initialize tracking variables
        self.portfolio_values = []
        self.balances = []
        self.stock_holdings = {ticker: [] for ticker in self.tickers}
        self.timestamps = []
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

        # Reset tracking variables
        self.portfolio_values = []
        self.balances = []
        self.stock_holdings = {ticker: [] for ticker in self.tickers}
        self.timestamps = []

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

        # Update tracking variables
        self.portfolio_values.append(total_value)
        self.balances.append(self.balance)
        self.timestamps.append(self.current_step)
        for ticker in self.tickers:
            self.stock_holdings[ticker].append(self.shares_held[ticker])

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

    def plot_portfolio_value(self):
        """Plot portfolio value and cash balance over time"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.timestamps, self.portfolio_values, label='Total Portfolio Value', color='blue')
        plt.plot(self.timestamps, self.balances, label='Cash Balance', color='green', alpha=0.6)
        
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Step')
        plt.ylabel('Value ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def plot_stock_holdings(self):
        """Plot number of shares held for each stock"""
        plt.figure(figsize=(12, 6))
        for ticker in self.tickers:
            plt.plot(self.timestamps, self.stock_holdings[ticker], label=f'{ticker} Shares', alpha=0.7)
        
        plt.title('Stock Holdings Over Time')
        plt.xlabel('Step')
        plt.ylabel('Number of Shares')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def plot_stock_prices(self):
        """Plot stock prices over the trading period"""
        plt.figure(figsize=(12, 6))
        for ticker in self.tickers:
            prices = self.df[ticker].iloc[self.timestamps]
            plt.plot(self.timestamps, prices, label=f'{ticker} Price', alpha=0.7)
        
        plt.title('Stock Prices Over Time')
        plt.xlabel('Step')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def plot_all_metrics(self):
        """Plot all metrics in a single figure with subplots"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        # Portfolio Value
        ax1.plot(self.timestamps, self.portfolio_values, label='Total Portfolio Value', color='blue')
        ax1.plot(self.timestamps, self.balances, label='Cash Balance', color='green', alpha=0.6)
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Stock Holdings
        for ticker in self.tickers:
            ax2.plot(self.timestamps, self.stock_holdings[ticker], label=f'{ticker} Shares', alpha=0.7)
        ax2.set_title('Stock Holdings Over Time')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Number of Shares')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Stock Prices
        for ticker in self.tickers:
            prices = self.df[ticker].iloc[self.timestamps]
            ax3.plot(self.timestamps, prices, label=f'{ticker} Price', alpha=0.7)
        ax3.set_title('Stock Prices Over Time')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Price ($)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def get_summary_stats(self):
        """Calculate and return summary statistics"""
        if not self.portfolio_values:
            return "No trading data available yet."
            
        initial_value = self.portfolio_values[0]
        final_value = self.portfolio_values[-1]
        returns = (final_value - initial_value) / initial_value * 100
        
        # Calculate daily returns
        daily_returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        
        # Calculate Sharpe Ratio (assuming risk-free rate of 0.01)
        risk_free_rate = 0.01
        excess_returns = daily_returns - risk_free_rate/252  # Daily risk-free rate
        sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
        
        # Calculate Maximum Drawdown
        peak = self.portfolio_values[0]
        max_drawdown = 0
        for value in self.portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return {
            'Initial Value': initial_value,
            'Final Value': final_value,
            'Total Return (%)': returns,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown (%)': max_drawdown * 100,
            'Total Profits': sum(self.profits)
        }