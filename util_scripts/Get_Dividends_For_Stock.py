from massive import RESTClient
import os
import pandas as pd

client = RESTClient(os.environ["Massive_API_Key"])


def get_stock_dividends(tickers: list[str], limit: int = 5000, sort: str = "ex_dividend_date.asc") -> pd.DataFrame:
    rows = []

    for row in client.list_stocks_dividends(
        limit=limit,
        sort=sort,
        ticker_any_of=tickers
    ):
        rows.append({
            "date": pd.to_datetime(row.ex_dividend_date).normalize(),
            "ticker": row.ticker,
            "dividend": row.cash_amount
        })

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    return df