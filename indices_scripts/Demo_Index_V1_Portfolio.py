import sys
import pandas as pd
from datetime import date
from pathlib import Path
import logging

# Configure logging for tracking loop progress
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- PATHING SETUP ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the specific function from your utilities
from util_scripts import compute_index_min_vol, compute_index_equal_weight

# =========================
# CONFIG
# =========================
START_DATE = date(2023, 6, 1)
END_DATE   = date(2026, 4, 30)

# Options: daily, weekly, monthly, every n week, every n month, quarterly, semi-annually, annually
REBALANCE_FREQ = "monthly" 

# Pointing to the root for the universe file to fetch unique sectors
RUSSELL_FILE = project_root / "russell3000_universe_monthly.csv"

# =========================
# EXECUTION
# =========================
if __name__ == "__main__":
    try:
        # Load universe file to extract the sectors dynamically
        logging.info(f"Loading universe file from: {RUSSELL_FILE}")
        holdings = pd.read_csv(RUSSELL_FILE)
        unique_sectors = holdings["sector"].dropna().unique().tolist()
        logging.info(f"Found {len(unique_sectors)} sectors to process: {unique_sectors}")
        
    except Exception as e:
        logging.error(f"Failed to read sector data from {RUSSELL_FILE}: {e}")
        sys.exit(1)

    # Loop through each individual sector
    for index_name in unique_sectors:
        logging.info(f"--- Starting Min-Vol optimization for sector: {index_name} ---")
        
        try:
            # Calling the function with the dynamically injected index_name
            output_path = compute_index_min_vol(
                name=index_name,
                start_date=START_DATE,
                end_date=END_DATE,
                rebalance_frequency=REBALANCE_FREQ
            )
            
            if output_path:
                logging.info(f"Successfully generated weights for {index_name} at: {output_path}")
            else:
                logging.warning(f"Optimization finished for {index_name}, but no output path was returned.")
                
        except Exception as e:
            # Captures optimization errors (like matrix inversions or rank deficiencies) 
            # for a specific sector without crashing the whole loop
            logging.error(f"Error during weight computation for sector '{index_name}': {e}")

    logging.info("Completed processing all sectors.")