import pandas as pd
import numpy as np
import re
from pathlib import Path
from scipy.optimize import minimize

def compute_index_min_vol(name, start_date, end_date, rebalance_frequency):
    # 1. Setup Paths
    clean_name = name.replace(" ", "_")
    data_dir = Path("indices_data") / f"{clean_name}_index_data"
    price_file = data_dir / f"{clean_name}_prices.csv"
    
    if not price_file.exists():
        raise FileNotFoundError(f"No price data found at {price_file}.")

    # 2. Load Data
    df = pd.read_csv(price_file)
    df['date'] = pd.to_datetime(df['date'])
    p_col = "adjusted_price" if "adjusted_price" in df.columns else "close"
    price_pivot = df.pivot(index='date', columns='ticker', values=p_col).ffill(limit=5)
    returns_pivot = price_pivot.pct_change()

    # 3. Frequency Parser
    freq_map = {"daily": "D", "weekly": "W", "monthly": "MS", "quarterly": "QS"}
    clean_freq = rebalance_frequency.lower().strip()
    alias = freq_map.get(clean_freq, "MS")
    if "every" in clean_freq:
        match = re.search(r'\d+', clean_freq)
        n = match.group() if match else "1"
        unit = "W" if "week" in clean_freq else "MS"
        alias = f"{n}{unit}"

    # 4. Timeline
    data_start = price_pivot.index.min()
    first_rb_possible = data_start + pd.DateOffset(years=1)
    full_schedule = pd.date_range(start=first_rb_possible, end=end_date, freq=alias)

    all_weights = []

    for rb_date in full_schedule:
        # Find the actual trading day
        actual_days = returns_pivot.index[returns_pivot.index >= rb_date]
        if actual_days.empty: 
            continue
        actual_rb_date = actual_days[0]
        
        # CRITICAL LAGGED LOGIC:
        # We want data ending the day BEFORE the rebalance date.
        data_end_date = returns_pivot.index[returns_pivot.index < actual_rb_date].max()
        lookback_start = data_end_date - pd.DateOffset(years=1)
        
        # Slice returns up to data_end_date (excluding actual_rb_date)
        raw_window = returns_pivot.loc[lookback_start : data_end_date].copy()
        
        # Data Cleaning
        clean_window = raw_window.dropna(axis=1, how='any')
        
        final_tickers = []
        for ticker in clean_window.columns:
            col_data = clean_window[ticker]
            if col_data.var() > 0 and not np.isnan(col_data.var()):
                final_tickers.append(ticker)
        
        period_returns = clean_window[final_tickers]
        n_tickers = len(final_tickers)

        if n_tickers < 10:
            continue

        cov_matrix = period_returns.cov().values * 252
        
        if np.any(np.isnan(cov_matrix)):
            continue

        # Optimization
        def objective(w): return w.T @ cov_matrix @ w
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
        bnds = tuple((0.0, 0.10) for _ in range(n_tickers))
        res = minimize(objective, np.array([1.0/n_tickers]*n_tickers), method='SLSQP', bounds=bnds, constraints=cons)

        if res.success:
            rb_df = pd.DataFrame({
                "rebalance_date": [actual_rb_date]*n_tickers, 
                "ticker": final_tickers, 
                "weight": res.x
            })
            all_weights.append(rb_df[rb_df['weight'] > 0.0001])

    if all_weights:
        final_df = pd.concat(all_weights)
        mask = (final_df['rebalance_date'] >= pd.to_datetime(start_date)) & \
               (final_df['rebalance_date'] <= pd.to_datetime(end_date))
        output_path = data_dir / f"{clean_name}_weights.csv"
        final_df.loc[mask].to_csv(output_path, index=False)
        print(f"Lagged weights saved to {output_path}")