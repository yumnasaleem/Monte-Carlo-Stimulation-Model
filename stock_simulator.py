import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import messagebox, ttk
from datetime import datetime, timedelta
import warnings

# Suppress specific warnings from yfinance
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='yfinance')

class MonteCarloSimulator:
    def __init__(self, ticker_symbol, start_date, end_date):
        self.ticker_symbol = ticker_symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.price_column = None

        fetched_data = self._fetch_data()

        if fetched_data is not None and not fetched_data.empty:
            # First, try to simplify column names if they are MultiIndex
            # This handles cases like ('Adj Close', 'AAPL') -> 'Adj Close'
            # and effectively drops the ticker level if it's a single ticker
            if isinstance(fetched_data.columns, pd.MultiIndex):
                # We expect the structure to be ('Column Name', 'Ticker Symbol')
                # So we take the first level for the column name
                fetched_data.columns = [col[0] for col in fetched_data.columns]
            
            # Now, check for 'Adj Close' or 'Close' in the simplified columns
            if 'Adj Close' in fetched_data.columns:
                self.price_column = 'Adj Close'
                self.data = fetched_data
            elif 'Close' in fetched_data.columns:
                self.price_column = 'Close'
                self.data = fetched_data
                messagebox.showwarning("Data Warning",
                                       f"'{self.price_column}' column used for {self.ticker_symbol} "
                                       "as 'Adj Close' was not found or adjusted. Results may vary slightly.")
            else:
                messagebox.showerror("Data Error", f"Neither 'Adj Close' nor 'Close' column found for {self.ticker_symbol} "
                                                   "after attempting to simplify column names. Cannot proceed.")
                self.log_returns = None
                self.mu = None
                self.sigma = None
                return

            # If we reached here, self.data and self.price_column are set
            # Ensure price_column has no NaN values at the beginning which can mess up pct_change
            if self.data[self.price_column].isnull().all():
                messagebox.showerror("Data Error", f"All values in the '{self.price_column}' column are NaN for {self.ticker_symbol}. Cannot proceed.")
                self.data = None
                self.log_returns = None
                self.mu = None
                self.sigma = None
                return

            self.log_returns = np.log(1 + self.data[self.price_column].pct_change()).dropna() # dropna important here
            if self.log_returns.empty:
                messagebox.showerror("Data Error", f"Insufficient data points for {self.ticker_symbol} to calculate log returns. Check date range or ticker.")
                self.data = None
                self.log_returns = None
                self.mu = None
                self.sigma = None
                return

            self.mu = self.log_returns.mean()
            self.sigma = self.log_returns.std()
        else:
            self.log_returns = None
            self.mu = None
            self.sigma = None

    def _fetch_data(self):
        try:
            # Explicitly set auto_adjust=False to try and get 'Adj Close' as a separate column
            # Also set actions=True to potentially force more columns
            stock_data = yf.download(self.ticker_symbol, start=self.start_date, end=self.end_date, auto_adjust=False, actions=True)
            if stock_data.empty:
                raise ValueError("No data fetched for the given ticker and date range.")
            return stock_data
        except Exception as e:
            messagebox.showerror("Data Fetch Error", f"Could not fetch data for {self.ticker_symbol}. Error: {e}")
            return None

    # The simulate method and StockApp class remain unchanged from the previous version
    # ... (rest of your code) ...
    def simulate(self, num_simulations=1000, num_days=252):
        if self.data is None or self.price_column is None or self.log_returns is None:
            return None, None

        S0 = self.data[self.price_column].iloc[-1]
        dt = 1 / num_days  # Assuming num_days is approx 1 year of trading days

        price_paths = np.zeros((num_days, num_simulations))
        price_paths[0] = S0

        for t in range(1, num_days):
            rand = np.random.standard_normal(num_simulations)
            drift = (self.mu - 0.5 * self.sigma**2) * dt
            diffusion = self.sigma * np.sqrt(dt) * rand
            price_paths[t] = price_paths[t-1] * np.exp(drift + diffusion)

        return price_paths, S0

class StockApp:
    def __init__(self, master):
        self.master = master
        master.title("Monte Carlo Stock Predictor")

        self.style = ttk.Style()
        self.style.theme_use('clam') # 'clam', 'alt', 'default', 'classic'

        # Input Frame
        self.input_frame = ttk.LabelFrame(master, text="Stock Details", padding="10")
        self.input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        ttk.Label(self.input_frame, text="Ticker Symbol:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.ticker_entry = ttk.Entry(self.input_frame, width=15)
        self.ticker_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.ticker_entry.insert(0, "AAPL") # Default value

        ttk.Label(self.input_frame, text="Start Date (YYYY-MM-DD):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.start_date_entry = ttk.Entry(self.input_frame, width=15)
        self.start_date_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.start_date_entry.insert(0, (datetime.now() - timedelta(days=365*3)).strftime("%Y-%m-%d")) # 3 years ago

        ttk.Label(self.input_frame, text="End Date (YYYY-MM-DD):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.end_date_entry = ttk.Entry(self.input_frame, width=15)
        self.end_date_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        self.end_date_entry.insert(0, datetime.now().strftime("%Y-%m-%d")) # Today

        ttk.Label(self.input_frame, text="Simulations:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.simulations_entry = ttk.Entry(self.input_frame, width=15)
        self.simulations_entry.grid(row=3, column=1, padx=5, pady=5, sticky="ew")
        self.simulations_entry.insert(0, "1000")

        ttk.Label(self.input_frame, text="Days to Predict:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.days_entry = ttk.Entry(self.input_frame, width=15)
        self.days_entry.grid(row=4, column=1, padx=5, pady=5, sticky="ew")
        self.days_entry.insert(0, "252") # Approx 1 trading year

        self.simulate_button = ttk.Button(self.input_frame, text="Run Monte Carlo Simulation", command=self.run_simulation)
        self.simulate_button.grid(row=5, column=0, columnspan=2, pady=10)

        # Output Frame
        self.output_frame = ttk.LabelFrame(master, text="Simulation Results", padding="10")
        self.output_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.output_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        self.info_label = ttk.Label(self.output_frame, text="Enter stock details and click 'Run Simulation'.")
        self.info_label.pack(pady=5)

        # Configure grid weights to make it responsive
        master.grid_rowconfigure(0, weight=1)
        master.grid_columnconfigure(1, weight=1)
        self.input_frame.grid_columnconfigure(1, weight=1)
        self.output_frame.grid_rowconfigure(0, weight=1)
        self.output_frame.grid_columnconfigure(0, weight=1)


    def run_simulation(self):
        ticker = self.ticker_entry.get().strip().upper()
        start_date_str = self.start_date_entry.get().strip()
        end_date_str = self.end_date_entry.get().strip()

        try:
            num_simulations = int(self.simulations_entry.get())
            num_days = int(self.days_entry.get())
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        except ValueError as e:
            messagebox.showerror("Input Error", f"Please check your input values. Error: {e}")
            return

        if not ticker:
            messagebox.showerror("Input Error", "Please enter a stock ticker symbol.")
            return

        simulator = MonteCarloSimulator(ticker, start_date, end_date)
        # Check if simulator.data was successfully populated and has a price column
        if simulator.data is None or simulator.price_column is None:
            return # Error message already shown by simulator's __init__ or _fetch_data

        price_paths, S0 = simulator.simulate(num_simulations, num_days)

        if price_paths is None:
            messagebox.showerror("Simulation Error", "Could not run simulation. Data might be insufficient or price column missing.")
            return

        # Clear previous plot
        self.ax.clear()

        # Plot historical data using the determined price_column
        self.ax.plot(simulator.data.index, simulator.data[simulator.price_column], label='Historical Prices', color='blue')

        # Generate future dates for prediction
        future_dates = pd.to_datetime([simulator.data.index[-1] + timedelta(days=i) for i in range(1, num_days + 1)])

        # Plot simulated paths
        for i in range(num_simulations):
            self.ax.plot(future_dates, price_paths[:, i], linewidth=0.5, alpha=0.1, color='red')

        self.ax.set_title(f'Monte Carlo Simulation for {ticker} Stock Price')
        self.ax.set_xlabel('Date')
        self.ax.set_ylabel('Adjusted Close Price') # Label remains 'Adj Close' as it's the ideal, even if using 'Close'
        self.ax.legend()
        self.ax.grid(True)
        self.fig.autofmt_xdate()
        self.canvas.draw()

        # Calculate and display predictions
        final_prices = price_paths[-1, :]
        mean_final_price = np.mean(final_prices)
        median_final_price = np.median(final_prices)
        min_final_price = np.min(final_prices)
        max_final_price = np.max(final_prices)

        self.info_label.config(text=f"Initial Price: ${S0:.2f}\n"
                                   f"Mean Predicted Price ({num_days} days): ${mean_final_price:.2f}\n"
                                   f"Median Predicted Price ({num_days} days): ${median_final_price:.2f}\n"
                                   f"Min Predicted Price: ${min_final_price:.2f}\n"
                                   f"Max Predicted Price: ${max_final_price:.2f}")

# Main part of the application
if __name__ == "__main__":
    root = tk.Tk()
    app = StockApp(root)
    root.mainloop()