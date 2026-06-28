import pandas as pd
import numpy as np
from pathlib import Path
import logging

def put_index_option_prices(df: pd.DataFrame, index_name: str) -> Path:
    """
    Saves processed historical options data to a dedicated directory structured 
    for options backtesting and engine consumption, enforcing strict column data types 
    and chronological sorting to prevent calculation engine failures.
    """
    # Format name and paths
    clean_name = index_name.replace(" ", "_")
    output_dir = Path("indices_data") / f"{clean_name}_index_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter for options-engine-required columns
    cols = [
        "date", 
        "ticker", 
        "expiration", 
        "strike", 
        "right", 
        "open", 
        "high", 
        "low", 
        "close", 
        "volume"
    ]
    
    # Extract only the specified columns that exist in the passed DataFrame
    df_save = df[[c for c in cols if c in df.columns]].copy()
    
    # ==============================================================================
    # ENGINE SAFE DATA TYPE ENFORCEMENT
    # ==============================================================================
    # 1. Critical Identifiers & Keys (Drop rows missing fundamental options mapping indices)
    critical_cols = ["date", "ticker", "expiration", "strike", "right"]
    df_save = df_save.dropna(subset=[c for c in critical_cols if c in df_save.columns])
    
    # Format Observation Date and Option Expiration Date to clean YYYY-MM-DD ISO strings
    if "date" in df_save.columns:
        df_save["date"] = pd.to_datetime(df_save["date"]).dt.strftime("%Y-%m-%d")
    if "expiration" in df_save.columns:
        df_save["expiration"] = pd.to_datetime(df_save["expiration"]).dt.strftime("%Y-%m-%d")
        
    # Standardize textual tags (Underlying Ticker and Put/Call Right flag) to clean uppercase
    if "ticker" in df_save.columns:
        df_save["ticker"] = df_save["ticker"].astype(str).str.strip().str.upper()
    if "right" in df_save.columns:
        df_save["right"] = df_save["right"].astype(str).str.strip().str.upper() # Maps 'P'/'C' or 'PUT'/'CALL'
        
    # 2. Strike Prices and Premium Matrix (Strict Floating Point Numbers)
    numeric_cols = ["strike", "open", "high", "low", "close"]
    for c in numeric_cols:
        if c in df_save.columns:
            df_save[c] = pd.to_numeric(df_save[c], errors="coerce").astype(float)
            
    # 3. Contract Trading Volume (Round first to handle data anomalies, then cast to strict Int64)
    if "volume" in df_save.columns:
        df_save["volume"] = (
            pd.to_numeric(df_save["volume"], errors="coerce")
            .fillna(0)
            .round()
            .astype("int64")
        )

    # ==============================================================================
    # DETERMINISTIC SORTING LAYER
    # ==============================================================================
    # Grouping by underlying ticker, then tracking across time, expiration, strike, and vertical right chain
    sort_order = ["ticker", "date", "expiration", "strike", "right"]
    active_sort_cols = [c for c in sort_order if c in df_save.columns]
    
    df_save = df_save.sort_values(by=active_sort_cols).reset_index(drop=True)
    
    # Save processed option contract pricing metrics
    price_file = output_dir / f"{clean_name}_option_prices.csv"
    df_save.to_csv(price_file, index=False)
    logging.info(f"Successfully formatted and staged option pricing matrix at: {price_file}")
        
    return output_dir