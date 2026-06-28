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

from util_scripts import put_index_weights, load_index_stock_prices

# ==============================================================================
# CONFIG & STRATEGY DESIGNATION
# ==============================================================================
INDEX_OFFICIAL_NAME = "Idiosyncratic_Quality"

# Strategy Allocation Hyperparameters (Shifted T-1 Execution)
LOOKBACK_WINDOW = 60      # Days to run factor model ordinary least squares regression
VOL_WINDOW = 20           # Days to track daily standard deviation for risk weighting
SELECTION_QUANTILE = 0.20 # Focus on bottom 20% for Longs, top 20% for Shorts
MAX_STOCK_WEIGHT = 0.08   # Single-stock concentration limit per leg (8%)

# ==============================================================================
# STRATEGY SIGNAL GENERATION (FACTOR DECOMPOSITION)
# ==============================================================================
def calculate_idiosyncratic_metrics(asset_returns, market_returns):
    """
    Decomposes asset returns against a market proxy proxy using OLS regression
    to extract the standard deviation of residuals (idiosyncratic risk).
    """
    if len(asset_returns) < LOOKBACK_WINDOW:
        return np.nan, np.nan
    
    # Run ordinary least squares linear regression: R_i = alpha + beta * R_m + e
    x = market_returns
    y = asset_returns
    
    # Fit line parameters
    beta, alpha = np.polyfit(x, y, 1)
    
    # Calculate residuals (error vector)
    predicted_y = beta * x + alpha
    residuals = y - predicted_y
    
    # Residual Variance is the variance of the idiosyncratic error term
    residual_variance = np.var(residuals, ddof=2) if len(residuals) > 2 else 0.0
    
    # Total historical risk standard deviation for inverse vol budgeting
    raw_daily_vol = np.std(y, ddof=1)
    
    return residual_variance, raw_daily_vol

def optimize_weights_with_cap(initial_weights, max_weight):
    """Iteratively enforces a maximum concentration ceiling on an array of weights."""
    weights = initial_weights.copy()
    num_assets = len(weights)
    
    for _ in range(10):
        overaged_mask = weights > max_weight
        if not overaged_mask.any():
            break
        excess_capital = np.sum(weights[overaged_mask] - max_weight)
        weights[overaged_mask] = max_weight
        
        underaged_mask = ~overaged_mask
        if underaged_mask.any():
            weights[underaged_mask] += excess_capital * (weights[underaged_mask] / np.sum(weights[underaged_mask]))
        else:
            weights = np.minimum(weights, 1.0 / num_assets)
            break
    return weights

# ==============================================================================
# PORTFOLIO BACKTEST EXECUTION
# ==============================================================================
try:
    logging.info(f"Fetching historical data matrix via utility loader for strategy execution...")
    # Targets your central data pool directory
    df = load_index_stock_prices(INDEX_OFFICIAL_NAME)
except FileNotFoundError as e:
    logging.error(e)
    sys.exit(1)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(by=["ticker", "date"]).reset_index(drop=True)

logging.info("Reshaping daily data streams into wide matrix form...")
wide_prices = df.pivot(index="date", columns="ticker", values="adjusted_close")
wide_prices = wide_prices.dropna(how="all", axis=1)

wide_returns = wide_prices.pct_change().dropna(how="all")
market_proxy_returns = wide_returns.mean(axis=1)

all_dates = wide_prices.index
fridays = all_dates[all_dates.dayofweek == 4]

portfolio_records = []
logging.info(f"Processing weekly rebalances across {len(fridays)} Friday horizons...")

for current_friday in fridays:
    try:
        friday_idx = wide_prices.index.get_loc(current_friday)
    except KeyError:
        continue
        
    historical_returns_slice = wide_returns.iloc[:friday_idx]
    historical_prices_slice = wide_prices.iloc[:friday_idx]
    
    if len(historical_returns_slice) < LOOKBACK_WINDOW:
        continue  
        
    returns_chunk = historical_returns_slice.iloc[-LOOKBACK_WINDOW:]
    market_chunk = market_proxy_returns.iloc[:friday_idx].iloc[-LOOKBACK_WINDOW:]
    
    residual_risks = {}
    valid_vols = {}
    
    for ticker in wide_prices.columns:
        asset_returns_series = returns_chunk[ticker].dropna()
        
        if len(asset_returns_series) == LOOKBACK_WINDOW:
            if historical_prices_slice[ticker].iloc[-1] > 0:
                res_var, total_vol = calculate_idiosyncratic_metrics(
                    asset_returns_series.values, 
                    market_chunk.values
                )
                if not np.isnan(res_var):
                    residual_risks[ticker] = res_var
                    valid_vols[ticker] = total_vol

    if not residual_risks:
        continue

    df_signals = pd.DataFrame({"residual_variance": residual_risks, "daily_vol": valid_vols}).dropna()
    if df_signals.empty:
        continue
        
    # ==============================================================================
    # TWO-TAILED ALPHA BASKET SELECTION
    # ==============================================================================
    # Long Basket: Bottom 20% of Residual Variance (Clean factor tracing)
    long_cutoff = df_signals["residual_variance"].quantile(SELECTION_QUANTILE)
    long_basket = df_signals[df_signals["residual_variance"] <= long_cutoff].copy()
    
    # Short Basket: Top 20% of Residual Variance (High speculative pricing noise)
    short_cutoff = df_signals["residual_variance"].quantile(1.0 - SELECTION_QUANTILE)
    short_basket = df_signals[df_signals["residual_variance"] >= short_cutoff].copy()
    
    if long_basket.empty or short_basket.empty:
        continue

    # ==============================================================================
    # RISK ALLOCATION & CAPPING LAYER
    # ==============================================================================
    # 1. Process Long Weights (Inverse Volatility)
    long_basket["inv_vol"] = 1.0 / long_basket["daily_vol"]
    long_vol_sum = long_basket["inv_vol"].sum()
    initial_long_weights = long_basket["inv_vol"].to_numpy() / long_vol_sum if long_vol_sum > 0 else np.ones(len(long_basket)) / len(long_basket)
    long_basket["weight"] = optimize_weights_with_cap(initial_long_weights, MAX_STOCK_WEIGHT)

    # 2. Process Short Weights (Inverse Volatility, assigned Negative Exposure)
    short_basket["inv_vol"] = 1.0 / short_basket["daily_vol"]
    short_vol_sum = short_basket["inv_vol"].sum()
    initial_short_weights = short_basket["inv_vol"].to_numpy() / short_vol_sum if short_vol_sum > 0 else np.ones(len(short_basket)) / len(short_basket)
    short_basket["weight"] = -optimize_weights_with_cap(initial_short_weights, MAX_STOCK_WEIGHT)

    # ==============================================================================
    # RECORD REBALANCE TRANSACTION VECTOR
    # ==============================================================================
    # Log Long Allocations
    for ticker, row in long_basket.iterrows():
        portfolio_records.append({
            "rebalance_date": current_friday.strftime("%Y-%m-%d"),
            "ticker": ticker,
            "weight": round(row["weight"], 6)
        })
        
    # Log Short Allocations
    for ticker, row in short_basket.iterrows():
        portfolio_records.append({
            "rebalance_date": current_friday.strftime("%Y-%m-%d"),
            "ticker": ticker,
            "weight": round(row["weight"], 6)
        })

# ==============================================================================
# MATRIX SAVE & EXPORT
# ==============================================================================
df_final_weights = pd.DataFrame(portfolio_records)

if not df_final_weights.empty:
    logging.info(f"Backtest engine completed. Transmitting dollar-neutral target data...")
    output_directory = put_index_weights(df=df_final_weights, index_name=INDEX_OFFICIAL_NAME)
    logging.info(f"Long-short residual variance spread vector successfully deployed to: {output_directory}")
else:
    logging.error("Allocation cycle resulted in an empty frame. Export process aborted.")