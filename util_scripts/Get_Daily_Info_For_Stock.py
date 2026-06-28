import pandas as pd
from datetime import date
from .Get_Dividends_For_Stock import get_stock_dividends
from .Get_EOD_Prices_For_Stock_And_Option import get_stock_eod
from .Get_Splits_For_Stock import get_stock_splits

def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.floor("D")
    return df

def load_prices(tickers, start_date, end_date):
    df = get_stock_eod(tickers, start_date, end_date)
    if df is None or df.empty: 
        return pd.DataFrame()
    
    df = normalize_dates(df.reset_index(drop=True))
    return df

def load_dividends(tickers):
    df = get_stock_dividends(tickers)
    if df is None or df.empty: 
        return pd.DataFrame(columns=["date", "ticker", "dividend"])
    
    df = normalize_dates(df)
    
    # Ensure lowercase matching
    df.columns = [c.lower() for c in df.columns]
    return df.groupby(["ticker", "date"], as_index=False)["dividend"].sum()

def load_splits(tickers):
    df = get_stock_splits(tickers)
    if df is None or df.empty: 
        return pd.DataFrame(columns=["date", "ticker", "adj_factor"])

    df = normalize_dates(df)
    out = []

    for ticker, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date", ascending=False).reset_index(drop=True)
        # Fallback to handle whatever split column name your split script uses
        split_col = "adjustment_factor" if "adjustment_factor" in g.columns else "adj_factor"
        g["factor"] = 1.0 / g[split_col]
        g["adj_factor"] = g["factor"].cumprod()
        out.append(g[["date", "ticker", "adj_factor"]])

    return pd.concat(out, ignore_index=True)

def get_stock_full_eod_data(tickers: list[str], start_date: date, end_date: date) -> pd.DataFrame:
    # 1. Load raw prices containing your exact columns (open, high, low, close, volume)
    prices = load_prices(tickers, start_date, end_date)
    if prices.empty: 
        return pd.DataFrame()

    dividends = load_dividends(tickers)
    splits = load_splits(tickers)

    final = []

    for ticker in tickers:
        p = prices[prices["ticker"] == ticker].copy()
        if p.empty:
            continue
            
        p = p.sort_values("date").reset_index(drop=True)

        # 2. Merge Dividends
        d = dividends[dividends["ticker"] == ticker][["date", "dividend"]]
        if not d.empty:
            p = p.merge(d, on="date", how="left")
        else:
            p["dividend"] = 0.0
        p["dividend"] = p["dividend"].fillna(0.0)

        # 3. Merge and compute Backward Split Factor
        s = splits[splits["ticker"] == ticker][["date", "adj_factor"]]
        if not s.empty:
            adj_map = s.set_index("date")["adj_factor"]
            p["adj_factor"] = p["date"].map(adj_map)
            p["adj_factor"] = p["adj_factor"].shift(-1)
            p["adj_factor"] = p["adj_factor"].bfill().fillna(1.0)
        else:
            p["adj_factor"] = 1.0

        # 4. Generate all raw and adjusted outputs
        p["adjusted_open"]   = (p["open"] * p["adj_factor"]).round(2)
        p["adjusted_high"]   = (p["high"] * p["adj_factor"]).round(2)
        p["adjusted_low"]    = (p["low"] * p["adj_factor"]).round(2)
        p["adjusted_close"]  = (p["close"] * p["adj_factor"]).round(2)
        
        # Capture and adjust volume if present
        if "volume" in p.columns:
            p["adjusted_volume"] = (p["volume"] * p["adj_factor"]).round(0).astype(int)
        else:
            p["volume"] = 0
            p["adjusted_volume"] = 0

        # Keep the core metrics clean and drop ticker-specific calculation noise if desired
        cols_to_keep = [
            "date", "ticker", "open", "high", "low", "close", "volume",
            "adjusted_open", "adjusted_high", "adjusted_low", "adjusted_close", "adjusted_volume",
            "dividend", "adj_factor"
        ]
        # Only slice the columns that actually exist to stay safe
        existing_cols = [c for c in cols_to_keep if c in p.columns]
        final.append(p[existing_cols])

    if not final:
        return pd.DataFrame()

    result = pd.concat(final, ignore_index=True)
    return result.sort_values(["ticker", "date"]).reset_index(drop=True)