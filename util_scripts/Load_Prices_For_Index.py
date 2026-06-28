import pandas as pd
from pathlib import Path
import logging

DATA_ROOT = Path("indices_data")

def load_index_stock_prices(index_name: str) -> pd.DataFrame:
    """Locates and loads the historical stock prices CSV for a given index."""
    clean_name = index_name.replace(" ", "_")
    target_file = DATA_ROOT / f"{clean_name}_index_data" / f"{clean_name}_stock_prices.csv"
    
    if not target_file.exists():
        error_msg = f"Missing stock file for '{index_name}' at: {target_file.resolve()}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)
        
    return pd.read_csv(target_file)


def load_index_option_prices(index_name: str) -> pd.DataFrame:
    """Locates and loads the historical option prices CSV for a given index."""
    clean_name = index_name.replace(" ", "_")
    target_file = DATA_ROOT / f"{clean_name}_index_data" / f"{clean_name}_option_prices.csv"
    
    if not target_file.exists():
        error_msg = f"Missing option file for '{index_name}' at: {target_file.resolve()}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)
        
    return pd.read_csv(target_file)