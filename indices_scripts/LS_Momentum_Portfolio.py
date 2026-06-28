import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- PATHING SETUP ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from util_scripts import put_index_weights, load_index_stock_prices

INDEX_OFFICIAL_NAME = "LS_Momentum"

# Hyperparameters (Using pure price-series inputs)
MOMENTUM_LOOKBACK = 252    # 1 Year of trading data to capture the trend
REVERSION_WINDOW = 21     # Exclude the most recent month to avoid short-term noise
VOL_LOOKBACK = 60          # Window for historical return volatility sizing
SELECTION_QUANTILE = 0.15 # Top/Bottom 15% of the available universe
MAX_STOCK_WEIGHT = 0.10   # Cap single-stock concentration risk

def optimize_weights_with_cap(initial_weights, max_weight, target_sum=1.0):
    weights = initial_weights.copy()
    if len(weights) == 0: return weights
    num_assets = len(weights)
    if max_weight * num_assets < target_sum:
        return np.full(num_assets, target_sum / num_assets)
        
    for _ in range(15):
        overaged_mask = weights > max_weight
        if not overaged_mask.any():
            break
        excess_capital = np.sum(weights[overaged_mask] - max_weight)
        weights[overaged_mask] = max_weight
        underaged_mask = ~overaged_mask
        if underaged_mask.any():
            weights[underaged_mask] += excess_capital * (weights[underaged_mask] / np.sum(weights[underaged_mask]))
            
    return (weights / np.sum(weights)) * target_sum

# Load standard OHLCV matrix
try:
    df = load_index_stock_prices(INDEX_OFFICIAL_NAME)
except FileNotFoundError as e:
    logging.error(e)
    sys.exit(1)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(by=["ticker", "date"]).reset_index(drop=True)

fridays = [d for d in sorted(df["date"].unique()) if pd.Timestamp(d).dayofweek == 4]
portfolio_records = []

logging.info(f"Processing cross-sectional pipeline over {len(fridays)} periods...")

for current_friday in fridays:
    df_historical = df[df["date"] <= current_friday].copy()
    active_signals = []
    
    for ticker, group in df_historical.groupby("ticker"):
        if len(group) < MOMENTUM_LOOKBACK + 1:
            continue
            
        group = group.sort_values(by="date").reset_index(drop=True)
        
        # Pull required historical close timestamps
        price_t_0 = group.iloc[-1]["adjusted_close"]
        price_t_21 = group.iloc[-REVERSION_WINDOW]["adjusted_close"]
        price_t_252 = group.iloc[-MOMENTUM_LOOKBACK]["adjusted_close"]
        
        # Math Check: Prevent division by zero errors on flat data
        if price_t_252 <= 0 or price_t_21 <= 0:
            continue
            
        # 12-1 Month Momentum Formula
        momentum_12_1 = (price_t_21 / price_t_252) - 1.0
        
        # Volatility Sizing Metric (Standard Deviation of Daily RETURNS)
        recent_returns = group.iloc[-VOL_LOOKBACK:]["adjusted_close"].pct_change().dropna()
        if len(recent_returns) < 10:
            continue
        return_vol = recent_returns.std()
        
        if pd.isna(return_vol) or return_vol == 0:
            continue
            
        active_signals.append({
            "ticker": ticker,
            "momentum": momentum_12_1,
            "return_vol": return_vol
        })
        
    if len(active_signals) < 10: # Ensure cross-sectional depth
        continue
        
    df_signals = pd.DataFrame(active_signals)
    
    # Cross-Sectional Cutoffs across the entire available price universe
    long_cutoff = df_signals["momentum"].quantile(1.0 - SELECTION_QUANTILE)
    short_cutoff = df_signals["momentum"].quantile(SELECTION_QUANTILE)
    
    long_basket = df_signals[df_signals["momentum"] >= long_cutoff].copy()
    short_basket = df_signals[df_signals["momentum"] <= short_cutoff].copy()
    
    if long_basket.empty or short_basket.empty:
        continue
        
    # ==========================================================================
    # VOLATILITY-PENALIZED SIZING (INVERSE RETURNS VOLATILITY)
    # ==========================================================================
    # Long Leg Allocation Sizing
    long_basket["inv_vol"] = 1.0 / long_basket["return_vol"]
    initial_long_w = long_basket["inv_vol"].to_numpy() / long_basket["inv_vol"].sum()
    final_long_w = optimize_weights_with_cap(initial_long_w, MAX_STOCK_WEIGHT, target_sum=1.0)
    
    # Short Leg Allocation Sizing
    short_basket["inv_vol"] = 1.0 / short_basket["return_vol"]
    initial_short_w = short_basket["inv_vol"].to_numpy() / short_basket["inv_vol"].sum()
    final_short_w = optimize_weights_with_cap(initial_short_w, MAX_STOCK_WEIGHT, target_sum=1.0)
    
    # Format and pack rows into memory
    for idx, row in enumerate(long_basket.itertuples()):
        portfolio_records.append({
            "rebalance_date": current_friday.strftime("%Y-%m-%d"),
            "ticker": row.ticker,
            "weight": round(final_long_w[idx], 6)
        })
        
    for idx, row in enumerate(short_basket.itertuples()):
        portfolio_records.append({
            "rebalance_date": current_friday.strftime("%Y-%m-%d"),
            "ticker": row.ticker,
            "weight": round(-final_short_w[idx], 6) # Negative weights for short entries
        })

# Save output data back out to database interface
df_final_weights = pd.DataFrame(portfolio_records)
if not df_final_weights.empty:
    output_directory = put_index_weights(df=df_final_weights, index_name=INDEX_OFFICIAL_NAME)
    logging.info(f"Clean pure-price 12-1M Momentum weights exported to: {output_directory}")
else:
    logging.error("Execution loop failed to yield any weights.")