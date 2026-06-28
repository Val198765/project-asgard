import sys
import pandas as pd
from datetime import date
from pathlib import Path
import logging

# Configure logging to see output clearly
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- PATHING SETUP ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from util_scripts import get_stock_full_eod_data, put_index_stock_prices

# =========================
# CONFIG
# =========================
START_DATE = date(2023, 6, 1)
END_DATE   = date(2026, 4, 30)

# Pointing to the root for the universe file
RUSSELL_FILE = project_root / "russell3000_universe_monthly.csv"

# =========================
# LOAD UNIVERSE & GET SECTORS
# =========================
# Load the universe once outside the loop for efficiency
holdings = pd.read_csv(RUSSELL_FILE)

# Dynamically extract all unique sectors, dropping any missing/null values
unique_sectors = holdings["sector"].dropna().unique().tolist()
logging.info(f"Found {len(unique_sectors)} sectors to process: {unique_sectors}")

# =========================
# LOOP THROUGH EACH SECTOR
# =========================
for index_name in unique_sectors:
    logging.info(f"Starting process for sector: {index_name}")
    
    # Identify the "Super-set" of tickers for this specific sector
    tickers = (
        holdings[
            (holdings["sector"] == index_name)
        ]["ticker"]
        .dropna()
        .unique()  # Handles duplicates from monthly time-series
        .tolist()[:50]
    )
    
    logging.info(f"Extracted {len(tickers)} unique tickers for {index_name}")
    
    if not tickers:
        logging.warning(f"No tickers found for sector {index_name}. Skipping.")
        continue

    # =========================
    # DATA EXECUTION
    # =========================
    try:
        # 1. Fetch bulk EOD data for the entire selection
        df_full = get_stock_full_eod_data(tickers, START_DATE, END_DATE)

        if df_full is not None and not df_full.empty:
            # 2. Stage the prices in the index folder
            put_index_stock_prices(df_full, index_name)
            logging.info(f"Successfully staged prices for {index_name}")
        else:
            logging.warning(f"No data returned from API/database for sector {index_name}")

    except Exception as e:
        logging.error(f"Selection/Download failed for {index_name}: {e}")

logging.info("Finished processing all sectors.")