import pandas as pd
from pathlib import Path
import logging

def put_index_stock_prices(df: pd.DataFrame, index_name: str) -> Path:
    """
    Saves processed historical stock data to a dedicated directory, enforcing strict 
    column data types and chronological sorting to prevent backtesting engine execution failures.
    """
    # Format name and paths
    clean_name = index_name.replace(" ", "_")
    output_dir = Path("indices_data") / f"{clean_name}_index_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter for engine-required columns
    cols = [
        "date", "ticker", "dividend", "adj_factor",
        "open", "high", "low", "close", "volume",
        "adjusted_open", "adjusted_high", "adjusted_low", "adjusted_close", "adjusted_volume"
    ]
    df_save = df[[c for c in cols if c in df.columns]].copy()
    
    # ==============================================================================
    # ENGINE SAFE DATA TYPE ENFORCEMENT
    # ==============================================================================
    # 1. Critical Identifiers (Drop rows with missing critical indices)
    df_save = df_save.dropna(subset=["date", "ticker"])
    
    # Convert dates to standard ISO strings (YYYY-MM-DD)
    df_save["date"] = pd.to_datetime(df_save["date"]).dt.strftime("%Y-%m-%d")
    
    # Force tickers to clean uppercase strings without whitespace
    df_save["ticker"] = df_save["ticker"].astype(str).str.strip().str.upper()
    
    # 2. Corporate Actions & Modifiers
    if "dividend" in df_save.columns:
        df_save["dividend"] = pd.to_numeric(df_save["dividend"], errors="coerce").fillna(0.0).astype(float)
    if "adj_factor" in df_save.columns:
        df_save["adj_factor"] = pd.to_numeric(df_save["adj_factor"], errors="coerce").fillna(1.0).astype(float)
        
    # 3. Raw and Adjusted Price Matrices (Strict Floating Point Numbers)
    price_cols = [
        "open", "high", "low", "close", 
        "adjusted_open", "adjusted_high", "adjusted_low", "adjusted_close"
    ]
    for c in price_cols:
        if c in df_save.columns:
            df_save[c] = pd.to_numeric(df_save[c], errors="coerce").astype(float)
            
    # 4. Volume Metric Formats (Strict Int64 to handle heavy institutional share sizes cleanly)
    vol_cols = ["volume", "adjusted_volume"]
    for c in vol_cols:
        if c in df_save.columns:
            df_save[c] = pd.to_numeric(df_save[c], errors="coerce").fillna(0).round().astype("int64")

    # ==============================================================================
    # DETERMINISTIC SORTING LAYER
    # ==============================================================================
    # Sorting first by ticker alphabetically, then date chronologically prevents rolling window shifting bugs
    df_save = df_save.sort_values(by=["ticker", "date"]).reset_index(drop=True)
    
    # Save Structured Clean Prices Matrix
    price_file = output_dir / f"{clean_name}_stock_prices.csv"
    df_save.to_csv(price_file, index=False)
    
    # Generate Equal Weight proxy file based on our new structured dataset
    tickers = df_save["ticker"].unique()
    if len(tickers) > 0:
        eq_weight = 1.0 / len(tickers)
        start_date = df_save["date"].min()
        
        weights_df = pd.DataFrame({
            "rebalance_date": [start_date] * len(tickers),
            "ticker": tickers,
            "weight": [round(eq_weight, 6)] * len(tickers)
        })
        
        # Enforce types on base weights frame out of habit
        weights_df["rebalance_date"] = pd.to_datetime(weights_df["rebalance_date"]).dt.strftime("%Y-%m-%d")
        weights_df["ticker"] = weights_df["ticker"].astype(str)
        weights_df["weight"] = weights_df["weight"].astype(float)
        
        weights_file = output_dir / f"{clean_name}_weights.csv"
        weights_df.to_csv(weights_file, index=False)

    return output_dir