import sys
import pandas as pd
from datetime import date
from pathlib import Path
import logging

# Configure logging to see execution progress clearly
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- PATHING SETUP ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Importing all 4 required utility scripts
from util_scripts import (
    get_stock_full_eod_data, 
    put_index_stock_prices,
    get_option_eod,
    put_index_option_prices
)

# ============================================================================
# CONFIGURATION
# ============================================================================
TARGET_TICKERS = ["AAPL", "MSFT"]
INDEX_NAME = "Mega_Cap_Tech"  # Creates 'Mega_Cap_Tech_index_data' directory

START_DATE = date(2023, 6, 1)
END_DATE   = date(2025, 12, 31)

logging.info(f"Initiating pipeline for {TARGET_TICKERS} from {START_DATE} to {END_DATE}")

# ============================================================================
# PHASE 1: HISTORICAL STOCK PRICES PIPELINE
# ============================================================================
logging.info("--- Starting Phase 1: Stock Price Extraction ---")
try:
    # 1. Fetch bulk underlying stock EOD records
    df_stocks = get_stock_full_eod_data(TARGET_TICKERS, START_DATE, END_DATE)

    if df_stocks is not None and not df_stocks.empty:
        # 2. Stage the prices and auto-generate the 50/50 equal weights file
        stock_dir = put_index_stock_prices(df_stocks, INDEX_NAME)
        logging.info(f"Successfully staged equity profiles in: {stock_dir}")
    else:
        logging.warning("No equity historical pricing data returned from the API database.")

except Exception as e:
    logging.error(f"Equity collection or pipeline saving failure: {e}")


# ============================================================================
# PHASE 2: HISTORICAL OPTION CHAINS PIPELINE (THETA DATA)
# ============================================================================
logging.info("--- Starting Phase 2: Options Chain Extraction ---")
try:
    # 3. Fetch bulk option history using our wildcard mechanism
    df_options = get_option_eod(TARGET_TICKERS, START_DATE, END_DATE)

    if df_options is not None and not df_options.empty:
        # 4. Stage option matrices cleanly into the same index folder 
        option_dir = put_index_option_prices(df_options, INDEX_NAME)
        logging.info(f"Successfully staged option contracts data matrix in: {option_dir}")
    else:
        logging.warning("No derivatives pricing data returned from the Theta Data API.")

except Exception as e:
    logging.error(f"Options chain collection or pipeline saving failure: {e}")

logging.info("Multi-asset script execution successfully complete.")