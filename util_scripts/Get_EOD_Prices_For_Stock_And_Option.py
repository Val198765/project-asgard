from thetadata import ThetaClient
# Import the specific error from thetadata
from thetadata.errors import NoDataFoundError 
from typing import cast
from datetime import date, timedelta
import pandas as pd
import os

client = ThetaClient(
    email=os.environ["Theta_Data_Username"],
    password=os.environ["Theta_Data_Password"],
    dataframe_type="pandas"
)

MAX_WINDOW_DAYS = 360

# Used only for options
class WildcardDate(str):
    """A string subclass that masquerades as a date by mimicking strftime."""
    def strftime(self, fmt: str) -> str:
        return self  # Seamlessly passes "*" back to the server when SDK runs .strftime()

# Used by both stocks and options
def _chunk_dates(start_date: date, end_date: date):
    current = start_date
    while current < end_date:
        next_date = min(current + timedelta(days=MAX_WINDOW_DAYS), end_date)
        yield current, next_date
        current = next_date

# For stocks here
def get_stock_eod(tickers: list[str], start_date: date, end_date: date) -> pd.DataFrame:
    all_data = []

    for ticker in tickers:
        all_chunks = []
        for start, end in _chunk_dates(start_date, end_date):
            # --- START OF CHANGES ---
            try:
                df = client.stock_history_eod(
                    symbol=ticker,
                    start_date=start,
                    end_date=end,
                )
            except NoDataFoundError:
                # Log that the ticker was skipped and move to the next chunk/ticker
                print(f"Skipping {ticker}: No data found for {start} to {end}")
                continue
            except Exception as e:
                # Log unexpected errors so you know if something else is wrong
                print(f"Unexpected error for {ticker}: {e}")
                continue
            # --- END OF CHANGES ---

            if df is None or df.empty or "created" not in df.columns:
                continue

            df["date"] = (pd.to_datetime(df["created"], utc=True).dt.tz_convert(None).dt.normalize())
            all_chunks.append(df)

        if not all_chunks:
            continue

        full = pd.concat(all_chunks, ignore_index=True)
        full = (
            full.sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )

        for col in ["open", "high", "low", "close", "bid", "ask"]:
            if col in full.columns:
                full[col] = full[col].astype(float).round(2)

        if "volume" in full.columns:
            full["volume"] = full["volume"].fillna(0).astype(int)

        full["ticker"] = ticker
        all_data.append(full)

    if not all_data:
        return pd.DataFrame()
    
    final_df = pd.concat(all_data, ignore_index=True)
    
    final_df = final_df[final_df["close"] > 0].copy()
    
    return (
        final_df.sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )

def get_option_eod(tickers: list[str], start_date: date, end_date: date) -> pd.DataFrame:
    all_data = []

    for ticker in tickers:
        all_chunks = []
        
        for start, end in _chunk_dates(start_date, end_date):
            try:
                # Masking the wildcard string as a date object to bypass strict library type hinting
                wildcard_expiration = cast(date, WildcardDate("*"))
                
                df = client.option_history_eod(
                    symbol=ticker,
                    expiration=wildcard_expiration,
                    start_date=start,
                    end_date=end,
                    strike="*",
                    right="both"
                )
            except NoDataFoundError:
                continue
            except Exception as e:
                print(f"Unexpected SDK execution error for {ticker} within chunk {start} to {end}: {e}")
                continue

            if df is None or df.empty or "created" not in df.columns:
                continue

            # Standardize date normalization matching the equity pipeline
            df["date"] = (pd.to_datetime(df["created"], utc=True).dt.tz_convert(None).dt.normalize())
            all_chunks.append(df)

        if not all_chunks:
            print(f"No historical records found for {ticker} across specified date ranges.")
            continue

        # Merge, clean and shape the data for this specific underlying ticker
        full = pd.concat(all_chunks, ignore_index=True)
        
        full = (
            full.sort_values("date")
            .drop_duplicates(subset=["date", "expiration", "strike", "right"], keep="last")
            .reset_index(drop=True)
        )

        for col in ["open", "high", "low", "close"]:
            if col in full.columns:
                full[col] = full[col].astype(float).round(2)

        if "volume" in full.columns:
            full["volume"] = full["volume"].fillna(0).astype(int)

        full["ticker"] = ticker
        
        if "strike" in full.columns:
            full["strike"] = full["strike"].astype(float).round(2)
        if "right" in full.columns:
            full["right"] = full["right"].astype(str).str.upper()
        if "expiration" in full.columns:
            full["expiration"] = pd.to_datetime(full["expiration"]).dt.date

        # Explicit production column ordering schema
        keep_cols = ["ticker", "date", "expiration", "strike", "right", "open", "high", "low", "close", "volume"]
        existing_cols = [col for col in keep_cols if col in full.columns]
        full = full[existing_cols]

        all_data.append(full)

    if not all_data:
        return pd.DataFrame()
    
    # Process final combined output structural operations
    final_df = pd.concat(all_data, ignore_index=True)
    final_df = final_df[final_df["close"] > 0].copy()
    
    return (
        final_df.sort_values(["date", "ticker", "expiration", "strike", "right"])
        .reset_index(drop=True)
    )