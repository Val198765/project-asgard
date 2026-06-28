# Data Schema Reference 📊

To ensure the **Gungnir** backtesting engine executes correctly, all data files must adhere to the following schemas. Any deviations in column naming or date formatting will result in execution errors.

## 1. Stock Prices Matrix
**File Path:** `indices_data/{index_name}_index_data/{index_name}_stock_prices.csv`

| Column | Type | Description | Required |
| :--- | :--- | :--- | :--- |
| `date` | String | Date in ISO format (`YYYY-MM-DD`) | **Yes** |
| `ticker` | String | Uppercase asset symbol (e.g., `AAPL`) | **Yes** |
| `close` | Float | Closing price for the day | **Yes** |
| `dividend` | Float | Cash dividend paid on this date (0.0 if none) | No |
| `adj_factor` | Float | Adjustment factor for splits/dividends (1.0 if none) | No |
| `open` | Float | Opening price | No |
| `high` | Float | Daily high price | No |
| `low` | Float | Daily low price | No |
| `volume` | Int64 | Number of shares traded | No |
| `adjusted_close` | Float | Close price adjusted for corporate actions | No |

---

## 2. Index Weights Matrix
**File Path:** `indices_data/{index_name}_index_data/{index_name}_weights.csv`

| Column | Type | Description | Required |
| :--- | :--- | :--- | :--- |
| `rebalance_date` | String | Date of portfolio rebalance (`YYYY-MM-DD`) | **Yes** |
| `ticker` | String | Uppercase asset symbol | **Yes** |
| `weight` | Float | Portfolio allocation (e.g., `0.05` for 5%) | **Yes** |

---

## 3. Risk-Free Rate (Benchmark)
**File Path:** `benchmark_data/risk_free_rate.csv`

| Column | Type | Description | Required |
| :--- | :--- | :--- | :--- |
| `date` | String | Date of the rate (`YYYY-MM-DD`) | **Yes** |
| `rate` | Float | Annualized rate in decimal (e.g., `0.045` for 4.5%) | **Yes** |

## 🛠 Formatting Tips
- **Dates:** Always use `YYYY-MM-DD`. The engine relies on string sorting for chronological order.
- **Tickers:** Always uppercase and trimmed of whitespace.
- **Weights:** Ensure the sum of weights for a single `rebalance_date` equals `1.0` for accurate performance tracking.
