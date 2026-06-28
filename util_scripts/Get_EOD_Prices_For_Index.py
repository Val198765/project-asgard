from thetadata import ThetaClient
from thetadata.errors import NoDataFoundError 
from datetime import date, timedelta
import pandas as pd
import os

client = ThetaClient(
    email=os.environ["Theta_Data_Username"],
    password=os.environ["Theta_Data_Password"],
    dataframe_type="pandas"
)

MAX_WINDOW_DAYS = 360

def _chunk_dates(start_date: date, end_date: date):
    current = start_date
    while current < end_date:
        next_date = min(current + timedelta(days=MAX_WINDOW_DAYS), end_date)
        yield current, next_date
        current = next_date

def get_index_eod(tickers: list[str], start_date: date, end_date: date) -> pd.DataFrame:
    all_data = []
    
    # Define the exact column schema you want to keep
    keep_cols = ["date", "ticker", "open", "high", "low", "close"]

    for ticker in tickers:
        all_chunks = []
        for start, end in _chunk_dates(start_date, end_date):
            df = None 
            try:
                df = client.index_history_eod(
                    symbol=ticker,
                    start_date=start,
                    end_date=end,
                )
            except NoDataFoundError:
                print(f"Skipping chunk for {ticker}: No data found for {start} to {end}")
                continue
            except Exception as e:
                print(f"Unexpected error for {ticker} from {start} to {end}: {e}")
                continue

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

        for col in ["open", "high", "low", "close"]:
            if col in full.columns:
                full[col] = full[col].astype(float).round(2)

        if "volume" in full.columns:
            full["volume"] = full["volume"].fillna(0).astype(int)
        else:
            # If the API ever omits volume for an index, initialize it to 0 to prevent a KeyError
            full["volume"] = 0

        full["ticker"] = ticker
        all_data.append(full)

    if not all_data:
        return pd.DataFrame(columns=keep_cols)
    
    final_df = pd.concat(all_data, ignore_index=True)
    final_df = final_df[final_df["close"] > 0].copy()
    
    # Sort and slice exclusively for your requested schema layout
    final_df = final_df.sort_values(["date", "ticker"]).reset_index(drop=True)
    
    # Ensure all specified columns exist in final_df before slicing to safely handle any edge cases
    final_df = final_df[[col for col in keep_cols if col in final_df.columns]]
    
    return final_df