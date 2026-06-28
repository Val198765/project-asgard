import pandas as pd
import numpy as np
from pathlib import Path
import logging

def put_index_weights(df: pd.DataFrame, index_name: str) -> Path:
    """
    Saves systematically calculated target portfolio weights to a dedicated directory,
    enforcing strict column schemas and deterministic sorting. Completely overwrites 
    any previous rebalance weights file for this specific index.
    """
    # Format name and target directory paths
    clean_name = index_name.replace(" ", "_")
    output_dir = Path("indices_data") / f"{clean_name}_index_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter for the specific columns required by the backtest engine
    cols = ["rebalance_date", "ticker", "weight"]
    df_save = df[[c for c in cols if c in df.columns]].copy()
    
    # ==============================================================================
    # ENGINE SAFE DATA TYPE ENFORCEMENT
    # ==============================================================================
    # 1. Critical Rebalance Identifiers (Drop rows missing key indexing components)
    critical_cols = ["rebalance_date", "ticker"]
    df_save = df_save.dropna(subset=[c for c in critical_cols if c in df_save.columns])
    
    # Format Rebalance Date to standard ISO strings (YYYY-MM-DD) for precise event-driven execution
    if "rebalance_date" in df_save.columns:
        df_save["rebalance_date"] = pd.to_datetime(df_save["rebalance_date"]).dt.strftime("%Y-%m-%d")
        
    # Standardize asset symbols to clean uppercase strings without whitespace noise
    if "ticker" in df_save.columns:
        df_save["ticker"] = df_save["ticker"].astype(str).str.strip().str.upper()
        # Drop any records that resolved to empty strings to avoid execution ghosts
        df_save = df_save[df_save["ticker"] != ""]
        
    # 2. Portfolio Allocation Scaling (Strict Floating Point Matrix)
    if "weight" in df_save.columns:
        # Force numeric, convert NaNs to 0% allocations, and round to 6 decimal places 
        # to remove floating-point tail noise while maintaining sub-basis-point precision
        df_save["weight"] = (
            pd.to_numeric(df_save["weight"], errors="coerce")
            .fillna(0.0)
            .round(6)
            .astype(float)
        )

    # ==============================================================================
    # DETERMINISTIC SORTING LAYER
    # ==============================================================================
    # Sorting chronologically by rebalance window, then alphabetically by ticker symbol.
    # This architecture guarantees that the engine reads allocations sequentially across time.
    ideal_sort_order = ["rebalance_date", "ticker"]
    active_sort_cols = [c for c in ideal_sort_order if c in df_save.columns]
    
    df_save = df_save.sort_values(by=active_sort_cols).reset_index(drop=True)
    
    # ==============================================================================
    # FILE EXPORT ZONE
    # ==============================================================================
    # Define the output file path
    weights_file = output_dir / f"{clean_name}_weights.csv"
    
    # Save the file (to_csv completely overwrites by default)
    df_save.to_csv(weights_file, index=False)
    logging.info(f"Successfully formatted, sorted, and overwrote index weights matrix at: {weights_file}")
        
    return output_dir