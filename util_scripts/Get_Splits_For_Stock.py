from massive import RESTClient
import os
import pandas as pd

client = RESTClient(os.environ["Massive_API_Key"])


def get_stock_splits(tickers: list[str], limit: int = 5000, sort: str = "execution_date.asc") -> pd.DataFrame:
    rows = []

    for row in client.list_stocks_splits(
        limit=limit,
        sort=sort,
        ticker_any_of=tickers
    ):
        rows.append({
            "date": pd.to_datetime(row.execution_date).normalize(),
            "ticker": row.ticker,
            "adjustment_factor": row.split_to / row.split_from
        })

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
        df = df.drop_duplicates(subset=["ticker", "date"], keep="last")

    return df