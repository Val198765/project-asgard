from thetadata import ThetaClient
import pandas as pd
import os
from pathlib import Path
from datetime import date

# Authentication mirroring existing scripts
client = ThetaClient(
    email=os.environ["Theta_Data_Username"],
    password=os.environ["Theta_Data_Password"],
    dataframe_type="pandas"
)

def get_rates_eod(symbol="SOFR", start_date=date(2020, 1, 1), end_date=date.today()):
    print(f"Fetching RF rates for {symbol} from {start_date} to {end_date}...")
    try:
        # Use the interest rate history endpoint
        df = client.interest_rate_history_eod(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )
        
        if df is None or df.empty:
            print(f"No data found for {symbol}")
            return None
        
        # Clean data: keep date and rate
        # Assuming columns are 'created' (date) and 'rate' (value) based on typical Theta Data response
        # Need to verify column names if this fails, but 'created' is used in Get_EOD_Prices_For_Index.py
        if "created" in df.columns:
            df["date"] = pd.to_datetime(df["created"], utc=True).dt.tz_convert(None).dt.normalize()
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        else:
            print(f"Unexpected columns in RF data: {df.columns}")
            return None

        # Standardize rate to decimal (e.g. 5.25 -> 0.0525)
        # Some APIs return as percentage, others as decimal. 
        # We check if the max value is > 1.0 to assume it's percentage.
        rate_col = "rate" if "rate" in df.columns else df.columns[0] # Fallback to first col if not named 'rate'
        if df[rate_col].max() > 1.0:
            df[rate_col] = df[rate_col] / 100.0
        
        df = df[["date", rate_col]].rename(columns={rate_col: "rate"})
        df["rate"] = df["rate"].round(4)
        df = df.sort_values("date").drop_duplicates(subset=["date"])
        
        return df

    except Exception as e:
        print(f"Error fetching RF rates: {e}")
        return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="SOFR")
    parser.add_argument("--start", type=str, default="2024-01-01")
    parser.add_argument("--end", type=str, default=None)
    args = parser.parse_args()
    
    start_dt = date.fromisoformat(args.start)
    end_dt = date.fromisoformat(args.end) if args.end else date.today()
    
    rf_df = get_rates_eod(symbol=args.symbol, start_date=start_dt, end_date=end_dt)
    if rf_df is not None:
        output_path = Path("benchmark_data/risk_free_rate.csv")
        rf_df.to_csv(output_path, index=False)
        print(f"Successfully saved RF rates to {output_path}")
    else:
        print("Failed to fetch RF rates.")
