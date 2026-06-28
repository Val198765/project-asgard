import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- PATHING SETUP ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the corrected utility functions from your repository package
from util_scripts import put_index_weights, load_index_stock_prices

# ==============================================================================
# CONFIG & STRATEGY DESIGNATION
# ==============================================================================
INDEX_OFFICIAL_NAME = "Quality_Weekly_Momentum"

# Strategy Allocation Hyperparameters
LOOKBACK_WINDOW = 60      # Days to evaluate log linear regression trend quality
VOL_WINDOW = 20           # Days to track daily standard deviation for weighting
SELECTION_QUANTILE = 0.75 # Focus on the top 25% strongest movers
MAX_STOCK_WEIGHT = 0.05   # Institutional single-stock concentration ceiling (5%)

# ==============================================================================
# STRATEGY SIGNAL GENERATION
# ==============================================================================
def calculate_momentum_score(y_series):
    """
    Calculates the Trend Score using the Linear Regression Slope of log prices 
    multiplied by the Coefficient of Determination (R-squared).
    """
    if len(y_series) < LOOKBACK_WINDOW:
        return np.nan
        
    x = np.arange(len(y_series))
    log_y = np.log(y_series)
    
    # Run ordinary least squares linear regression
    slope, intercept = np.polyfit(x, log_y, 1)
    
    # Compute R-squared (Coefficient of Determination)
    y_pred = slope * x + intercept
    residuals = log_y - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Return the quality-adjusted trend slope
    return slope * r_squared

# ==============================================================================
# PORTFOLIO BACKTEST EXECUTION
# ==============================================================================
try:
    logging.info(f"Fetching historical data matrix via utility loader for: '{INDEX_OFFICIAL_NAME}'")
    # Updated to your custom 'load' script convention
    df = load_index_stock_prices(INDEX_OFFICIAL_NAME)
except FileNotFoundError as e:
    logging.error(e)
    sys.exit(1)

# Enforce clean date formats and sort chronologically
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(by=["ticker", "date"]).reset_index(drop=True)

# Reshape raw data streams into a wide matrix (Rows: Dates, Columns: Tickers)
logging.info("Reshaping daily data streams into wide matrix form...")
wide_prices = df.pivot(index="date", columns="ticker", values="adjusted_close")
wide_prices = wide_prices.dropna(how="all", axis=1)

# Isolate all trading dates that land on Fridays for weekly rebalancing
all_dates = wide_prices.index
fridays = all_dates[all_dates.dayofweek == 4]

portfolio_records = []
logging.info(f"Processing weekly rebalances across {len(fridays)} Friday horizons...")

for current_friday in fridays:
    # Find the integer position of the current Friday in the wide matrix
    friday_idx = wide_prices.index.get_loc(current_friday)
    
    # Slice historical data up to Thursday (T-1), excluding the Friday row
    historical_slice = wide_prices.iloc[:friday_idx]
    
    if len(historical_slice) < max(LOOKBACK_WINDOW, VOL_WINDOW):
        continue  # Skip initial periods until warm-up windows are filled
        
    # Isolate lookback windows for calculations
    momentum_chunk = historical_slice.iloc[-LOOKBACK_WINDOW:]
    volatility_chunk = historical_slice.iloc[-VOL_WINDOW:]
    
    scores = {}
    valid_vols = {}
    
    # Calculate indicators for each active ticker in the universe
    for ticker in wide_prices.columns:
        prices_series = momentum_chunk[ticker].dropna()
        vol_series = volatility_chunk[ticker].dropna()
        
        if len(prices_series) == LOOKBACK_WINDOW and len(vol_series) == VOL_WINDOW:
            if prices_series.iloc[-1] > 0 and not vol_series.is_unique:
                scores[ticker] = calculate_momentum_score(prices_series.values)
                # Compute daily percentage returns standard deviation
                daily_returns = vol_series.pct_change().dropna()
                valid_vols[ticker] = daily_returns.std()

    if not scores:
        continue

    # Clean signal framing
    df_signals = pd.DataFrame({"trend_score": scores, "daily_vol": valid_vols}).dropna()
    if df_signals.empty:
        continue
        
    # Isolate top 15% performing alpha basket
    cutoff_score = df_signals["trend_score"].quantile(SELECTION_QUANTILE)
    long_basket = df_signals[df_signals["trend_score"] >= cutoff_score].copy()
    
    if long_basket.empty:
        continue

    # Allocate Inverse-Volatility weights: w_i = (1 / sigma_i)
    long_basket["inv_vol"] = 1.0 / long_basket["daily_vol"]
    inv_vol_sum = long_basket["inv_vol"].sum()
    
    if inv_vol_sum > 0:
        long_basket["weight"] = long_basket["inv_vol"] / inv_vol_sum
    else:
        long_basket["weight"] = 1.0 / len(long_basket)

    # --- FIXED COPTIMIZATION LOOP ---
    # .to_numpy().copy() unlinks the data from the read-only DataFrame view
    weights = long_basket["weight"].to_numpy().copy()
    num_assets = len(weights)
    
    for _ in range(10):
        overaged_mask = weights > MAX_STOCK_WEIGHT
        if not overaged_mask.any():
            break
        excess_capital = np.sum(weights[overaged_mask] - MAX_STOCK_WEIGHT)
        weights[overaged_mask] = MAX_STOCK_WEIGHT
        
        underaged_mask = ~overaged_mask
        if underaged_mask.any():
            weights[underaged_mask] += excess_capital * (weights[underaged_mask] / np.sum(weights[underaged_mask]))
        else:
            weights = np.minimum(weights, 1.0 / num_assets)
            break

    # Safely write the re-allocated, mutable weights array back to the frame
    long_basket["weight"] = weights

    # Record final target weights schema
    for ticker, row in long_basket.iterrows():
        portfolio_records.append({
            "rebalance_date": current_friday.strftime("%Y-%m-%d"),
            "ticker": ticker,
            "weight": round(row["weight"], 6)
        })

# ==============================================================================
# MATRIX SAVE & EXPORT VIA UTIL FUNCTION
# ==============================================================================
df_final_weights = pd.DataFrame(portfolio_records)

if not df_final_weights.empty:
    logging.info(f"Backtest engine completed. Transmitting target data to pipeline...")
    
    # Hand off weights DataFrame to utility writer to format and overwrite old logs
    output_directory = put_index_weights(
        df=df_final_weights, 
        index_name=INDEX_OFFICIAL_NAME
    )
    
    logging.info(f"Target index metrics successfully deployed to: {output_directory}")
else:
    logging.error("Allocation cycle resulted in an empty frame. Export process aborted.")