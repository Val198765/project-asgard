import sys
import pandas as pd
from datetime import date
from pathlib import Path
import logging

# Configure logging to see download progress
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- PATHING SETUP ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from util_scripts import get_stock_full_eod_data, put_index_stock_prices

# ==============================================================================
# CONFIG & DESIGNATION
# ==============================================================================
INDEX_OFFICIAL_NAME = "Quality_Weekly_Momentum"

START_DATE = date(2023, 6, 1)
END_DATE   = date(2026, 5, 16)

# Institutional Size Hurdle ($5 Million)
MARKET_VALUE_HURDLE = 5_000_000 

# Path to the source monthly universe mapping file
RUSSELL_FILE = project_root / "russell3000_universe_monthly.csv"

# ==============================================================================
# UNIVERSE AGGREGATION & CLEANING
# ==============================================================================
logging.info(f"Loading Russell 3000 historical universe tracker: {RUSSELL_FILE.name}")
holdings = pd.read_csv(RUSSELL_FILE)

# Clean the string-based market values (stripping out thousands-separator commas)
if "market_value" in holdings.columns:
    logging.info("Cleaning and formatting 'market_value' column strings into numeric values...")
    
    # Cast to string, strip commas, convert to float, handle errors gracefully
    holdings["market_value"] = (
        holdings["market_value"]
        .astype(str)
        .str.replace(",", "", regex=False)
    )
    holdings["market_value"] = pd.to_numeric(holdings["market_value"], errors="coerce")
else:
    raise KeyError("Could not locate 'market_value' column. Please check the schema of your CSV file.")

# Extract the entire investable universe that meets the asset size requirement
eligible_mask = holdings["market_value"] >= MARKET_VALUE_HURDLE
tickers = holdings[eligible_mask]["ticker"].dropna().unique().tolist()

logging.info(f"Total unique tickers meeting the $15MM institutional floor: {len(tickers)}")

if not tickers:
    logging.error("No tickers survived the size filter. Execution halted.")
    sys.exit(1)

# ==============================================================================
# DATA CAPTURE & STAGING
# ==============================================================================
try:
    logging.info(f"Initiating bulk EOD data pull for {INDEX_OFFICIAL_NAME}...")
    logging.info(f"Date Range: {START_DATE} to {END_DATE}")
    
    # Fetch unified historical daily pricing data frame
    df_master_prices = get_stock_full_eod_data(tickers, START_DATE, END_DATE)

    if df_master_prices is not None and not df_master_prices.empty:
        # Stage the master dataset into the index folder under the official name
        put_index_stock_prices(df_master_prices, INDEX_OFFICIAL_NAME)
        logging.info(f"Successfully staged master data stream for {INDEX_OFFICIAL_NAME}.")
        logging.info(f"Staged DataFrame Shape: {df_master_prices.shape}")
    else:
        logging.error("Data vendor returned an empty dataset or failed connection.")

except Exception as e:
    logging.error(f"Critical error during pricing staging phase for {INDEX_OFFICIAL_NAME}: {e}")

logging.info("Data staging phase finalized.")