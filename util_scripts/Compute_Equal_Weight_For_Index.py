import pandas as pd
from pathlib import Path
from pandas.tseries.frequencies import to_offset

def compute_index_equal_weight(name, start_date, end_date, rebalance_frequency):
    """
    Computes equal-weight rebalancing dates and weights based on available tickers.
    
    Args:
        name (str): Name of the index.
        start_date (str or pd.Timestamp): Start of the backtest.
        end_date (str or pd.Timestamp): End of the backtest.
        rebalance_frequency (str): Frequency (e.g., 'D', 'W', 'M', '3W', '2M', 'Q', '6M', 'A').
    """
    # 1. Setup Paths (Mirroring your previous logic)
    clean_name = name.replace(" ", "_")
    data_dir = Path("indices_data") / f"{clean_name}_index_data"
    price_file = data_dir / f"{clean_name}_prices.csv"
    
    if not price_file.exists():
        raise FileNotFoundError(f"No price data found at {price_file}. Please run put_index_prices first.")

    # 2. Load and Filter Data
    df = pd.read_csv(price_file)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter for the date range
    mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
    df_period = df.loc[mask].copy()
    
    if df_period.empty:
        print("No data found for the specified date range.")
        return None

    # 3. Define Rebalance Dates
    # Map friendly names to Pandas Offset Aliases
    freq_map = {
        "daily": "D",
        "weekly": "W",
        "monthly": "MS", # Month Start
        "quarterly": "QS", 
        "semi-annually": "6MS",
        "annually": "YS"
    }
    
    # Handle "every n week/month" or use the direct alias
    alias = freq_map.get(rebalance_frequency.lower(), rebalance_frequency)
    
    # Generate the schedule
    all_rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=alias)
    
    # 4. Compute Weights for each Rebalance Event
    all_weights = []
    
    for rb_date in all_rebalance_dates:
        # Find the actual trading day (in case rebalance date is a weekend/holiday)
        # We look for the first available date in the DF that is >= the scheduled date
        available_dates = df_period[df_period['date'] >= rb_date]['date']
        if available_dates.empty:
            continue
            
        actual_rb_date = available_dates.min()
        
        # Get tickers that have a price on this specific day
        active_tickers = df_period[df_period['date'] == actual_rb_date]['ticker'].unique()
        
        if len(active_tickers) > 0:
            weight = 1.0 / len(active_tickers)
            rb_df = pd.DataFrame({
                "rebalance_date": [actual_rb_date] * len(active_tickers),
                "ticker": active_tickers,
                "weight": [weight] * len(active_tickers)
            })
            all_weights.append(rb_df)

    # 5. Save Output
    if all_weights:
        final_weights_df = pd.concat(all_weights).drop_duplicates()
        
        # Consistent naming convention: [name]_index_data.csv
        output_file = data_dir / f"{clean_name}_weights.csv"
        final_weights_df.to_csv(output_file, index=False)
        
        print(f"Success! Equal weight file created at: {output_file}")
        return output_file
    else:
        print("No rebalance weights could be calculated.")
        return None

# --- Example Call ---
# compute_equal_weight("Tech Giant", "2023-01-01", "2023-12-31", "monthly")