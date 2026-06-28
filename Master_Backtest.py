import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from weasyprint import HTML, CSS

# ==============================================================================
# CONFIGURATION & SETTINGS
# ==============================================================================
TRADING_DAYS = 252
START_VALUE = 10000.0  # Normalized Institutional Base Value
RF_RATE_DEFAULT = 0.03       # Fallback Risk-Free Rate
RF_DATA_PATH = Path("benchmark_data/risk_free_rate.csv")
SHORT_BORROW_FEE = 0.0050    # 50bps Annualized
CASH_BORROW_SPREAD = 0.02    # 200bps over SOFR
CASH_INTEREST_RATE_PCT = 0.90 # Earn 90% of SOFR on positive cash
COLLATERAL_REQ = 1.02        # 102% Collateral Requirement
OUTPUT_DIR = Path("backtest_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Dedicated Benchmark Configuration (Strict Path Alignment)
BENCHMARK_PATH = Path("benchmark_data/Russell_3000_Benchmark.csv")

# Corporate Branding Assets
LOGO_PATH = Path("none")
THETA_LOGO_PATH = Path("none")
MAIN_COLOR = "#0A192F"     # Deep Institutional Navy
BORDER_COLOR = "#CBD5E0"   # Light Gray Border

# ==============================================================================
# QUANTITATIVE FINANCIAL METRICS ENGINE
# ==============================================================================
def compute_drawdown(equity):
    peak = equity.cummax()
    return (equity / peak - 1.0).fillna(0)

def sharpe(r):
    if len(r) < 2 or r.std() == 0: 
        return 0
    return np.sqrt(TRADING_DAYS) * r.mean() / r.std()

def sortino(r):
    downside = r[r < 0].std()
    if len(r) < 2 or downside == 0: 
        return 0
    return np.sqrt(TRADING_DAYS) * r.mean() / downside

def CAGR(equity):
    if len(equity) < 2: 
        return 0
    years = len(equity) / TRADING_DAYS
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else 0

def rolling_metric(r, func, window=252):
    return r.rolling(window).apply(func).fillna(0)

# ==============================================================================
# INITIALIZE BENCHMARK DATA SYSTEM
# ==============================================================================
if not BENCHMARK_PATH.exists():
    print(f"Error: Core benchmark file not found at '{BENCHMARK_PATH}'.")
    exit()

# Load and prepare master benchmark reference series
bench_raw = pd.read_csv(BENCHMARK_PATH)
bench_date_col = None
for col in bench_raw.columns:
    if "date" in col.lower():
        bench_date_col = col
        break

if bench_date_col:
    bench_raw[bench_date_col] = pd.to_datetime(bench_raw[bench_date_col])
    bench_raw = bench_raw.set_index(bench_date_col).sort_index()
else:
    bench_raw = pd.read_csv(BENCHMARK_PATH, parse_dates=True, index_col=0).sort_index()

# Extract closest column matches for price/close data
bench_price_col = bench_raw.columns[0]
for col in bench_raw.columns:
    if any(k in col.lower() for k in ["close", "adj", "price", "val"]):
        bench_price_col = col
        break
bench_master_series = bench_raw[bench_price_col]

# ==============================================================================
# MAIN BACKTEST PROCESSING ENGINE
# ==============================================================================
root_path = Path("indices_data")
if not root_path.exists():
    print("Error: 'indices_data' folder not found.")
    exit()

for folder in root_path.rglob("*_index_data"):
    for pf in folder.glob("*_stock_prices.csv"):
        name = pf.name.replace("_stock_prices.csv", "")
        wf = folder / f"{name}_weights.csv"

        if not wf.exists():
            continue

        # Format and institutionalize index naming conventions
        clean_name = name.replace("_", " ").strip().upper()
        full_index_name = f"Asgard {clean_name}"

        # Load and sort operational files
        weights = pd.read_csv(wf, parse_dates=["rebalance_date"]).sort_values("rebalance_date")
        df_raw = pd.read_csv(pf, parse_dates=["date"])

        # Pivot & Sanitize Raw Data Matrices
        prices = df_raw.pivot(index="date", columns="ticker", values="adjusted_close").sort_index()
        dividends = df_raw.pivot(index="date", columns="ticker", values="dividend").fillna(0)

        # Build structural asset life cycle timelines
        last_trade = {t: prices[t].last_valid_index() for t in prices.columns}
        weight_map = {d: g.set_index("ticker")["weight"].to_dict() for d, g in weights.groupby("rebalance_date")}
        rebalance_dates = sorted(weight_map.keys())

        # Dynamic Rebalancing Frequency Detection Core
        if len(rebalance_dates) > 1:
            days_intervals = pd.Series(rebalance_dates).diff().dt.days.dropna()
            avg_interval = days_intervals.mean()
            if 5 <= avg_interval <= 9: rebal_freq_str = "Weekly"
            elif 26 <= avg_interval <= 35: rebal_freq_str = "Monthly"
            elif 80 <= avg_interval <= 100: rebal_freq_str = "Quarterly"
            elif 160 <= avg_interval <= 200: rebal_freq_str = "Semi-Annually"
            elif 330 <= avg_interval <= 380: rebal_freq_str = "Annually"
            else: rebal_freq_str = f"Dynamic ({int(round(avg_interval))} Days)"
        else:
            rebal_freq_str = "Static Inception Only"

        # Align timeline boundaries strictly to the first date found in weights file
        start_date = rebalance_dates[0]
        inception_date_str = start_date.strftime('%Y-%m-%d')
        
        prices = prices.loc[prices.index >= start_date]
        dividends = dividends.loc[dividends.index >= start_date]
        dates = prices.index

        # Load and align dynamic Risk-Free Rate for excess return calculations
        if RF_DATA_PATH.exists():
            rf_raw = pd.read_csv(RF_DATA_PATH, parse_dates=["date"]).set_index("date")
            daily_rf_series = rf_raw["rate"].reindex(dates).ffill().fillna(RF_RATE_DEFAULT)
        else:
            daily_rf_series = pd.Series(RF_RATE_DEFAULT, index=dates)
        
        # Convert annualized RF rate to daily equivalent
        daily_rf_dec = daily_rf_series / TRADING_DAYS
        
        print(f"Processing {full_index_name} Production Pipeline...")

        equity = pd.Series(index=dates, dtype=float)
        portfolio_cash = pd.Series(index=dates, dtype=float)
        portfolio_div_yield = pd.Series(0.0, index=dates)
        daily_tx_costs = pd.Series(0.0, index=dates)
        daily_tx_pct = pd.Series(0.0, index=dates)  # Tracking percentages for precise trailing TER
        
        hist_weights = pd.DataFrame(0.0, index=dates, columns=prices.columns.tolist() + ["CASH"])
        hist_contribs = pd.DataFrame(0.0, index=dates, columns=prices.columns.tolist() + ["CASH"])
        
        # State Execution Memory
        shares = {}
        max_revolver_draw_pct = 0.0
        
        # Execute Target Rebalance Sequence (T0) - Incorporating Inception Friction
        first_weights = weight_map[start_date]
        first_prices = prices.loc[dates[0]]
        
        # Inception portfolio goes from all cash to target assets
        total_weight_change_t0 = sum(abs(w) for w in first_weights.values())
        tx_cost_t0 = 0.0003 * total_weight_change_t0 * START_VALUE
        daily_tx_costs.iloc[0] = tx_cost_t0
        daily_tx_pct.iloc[0] = 0.0003 * total_weight_change_t0
        
        nav = START_VALUE - tx_cost_t0
        equity.iloc[0] = nav
        
        long_exposure = 0.0
        short_proceeds = 0.0

        for t, w in first_weights.items():
            if t in first_prices and pd.notnull(first_prices[t]) and first_prices[t] > 0 and w != 0:
                shares[t] = (nav * w) / first_prices[t]
                hist_weights.at[dates[0], t] = w
                if w > 0:
                    long_exposure += (nav * w)
                else:
                    short_proceeds += abs(nav * w)
        
        cash = nav - long_exposure + short_proceeds
        portfolio_cash.iloc[0] = cash
        hist_weights.at[dates[0], "CASH"] = cash / nav

        # ======================================================================
        # CHRONOLOGICAL SIMULATION LOOP
        # ======================================================================
        for i in range(1, len(dates)):
            date = dates[i]
            prev_date = dates[i - 1]
            prev_nav = equity.iloc[i - 1]

            row_prices = prices.loc[date]
            prev_row_prices = prices.loc[prev_date]
            row_divs = dividends.loc[date]

            # Unified Asset Processing Loop (Dividends, Delistings, MTM, and Exposure)
            div_received = 0.0
            cash_from_delist = 0.0
            equity_value = 0.0
            short_exposure = 0.0
            shares_to_purge = []

            for t, sh in shares.items():
                if sh == 0: continue
                
                # 1. Process Dividends
                if t in row_divs:
                    div_received += sh * row_divs[t]
                
                # 2. Process Asset End-Of-Life Liquidation
                if date == last_trade.get(t):
                    p_delist = row_prices.get(t, np.nan)
                    if pd.isnull(p_delist) or np.isnan(p_delist):
                        p_delist = prev_row_prices.get(t, 0.0)
                    cash_from_delist += sh * p_delist
                    shares_to_purge.append(t)
                    continue

                # 3. Mark to Market & Exposure Calculation
                p_curr = row_prices.get(t, np.nan)
                p_prev = prev_row_prices.get(t, np.nan)
                
                if pd.notnull(p_curr) and not np.isnan(p_curr):
                    equity_value += sh * p_curr
                    if sh < 0:
                        short_exposure += abs(sh * p_curr)
                    
                    p_diff = p_curr - p_prev if pd.notnull(p_prev) and not np.isnan(p_prev) else 0.0
                    div_factor = row_divs.get(t, 0.0)
                    asset_dollar_return = (sh * p_diff) + (sh * div_factor)
                    hist_contribs.at[date, t] = asset_dollar_return / prev_nav if prev_nav > 0 else 0

            # Settle Cash Flows
            portfolio_div_yield.iloc[i] = div_received / prev_nav if prev_nav > 0 else 0
            cash += (div_received + cash_from_delist)

            # Purge delisted assets
            for t in shares_to_purge:
                if t in shares: del shares[t]

            # Process Daily Financing Costs (Short Borrowing, Cash Borrowing, and Cash Interest)
            short_fee = short_exposure * (SHORT_BORROW_FEE / TRADING_DAYS)
            cash_finance_cost = 0.0
            
            current_sofr = daily_rf_series.iloc[i]
            if cash < 0:
                # Pay SOFR + Spread on negative cash
                cash_finance_cost = abs(cash) * ((current_sofr + CASH_BORROW_SPREAD) / TRADING_DAYS)
            else:
                # Earn % of SOFR on positive cash
                cash_finance_cost = - (cash * (current_sofr * CASH_INTEREST_RATE_PCT / TRADING_DAYS))
            
            # Revolver Logic: Handle Collateral Gap (102% Requirement)
            # Collateral = Cash + Longs. If < 102% of Shorts, draw on Revolver.
            long_exposure = 0.0
            for t, sh in shares.items():
                if sh > 0:
                    long_exposure += sh * row_prices.get(t, 0.0)
            
            collateral_available = cash + long_exposure
            collateral_required = short_exposure * COLLATERAL_REQ
            revolver_draw = max(0, collateral_required - collateral_available)
            revolver_fee = revolver_draw * ((current_sofr + CASH_BORROW_SPREAD) / TRADING_DAYS)
            
            # Track Maximum Revolver Usage as % of NAV
            if nav > 0:
                max_revolver_draw_pct = max(max_revolver_draw_pct, revolver_draw / nav)
            
            cash -= (short_fee + cash_finance_cost + revolver_fee)

            # Structural Ledger Settlement Execution
            nav = equity_value + cash

            # Apply Daily Management Fee (MER) to NAV for "Net" performance tracking
            annual_mer = 0.0
            if "weekly" in rebal_freq_str.lower(): annual_mer = 0.0070
            elif "monthly" in rebal_freq_str.lower(): annual_mer = 0.0038
            elif "quarterly" in rebal_freq_str.lower(): annual_mer = 0.0020
            elif "semi-annually" in rebal_freq_str.lower(): annual_mer = 0.0011
            elif "annually" in rebal_freq_str.lower(): annual_mer = 0.0006
            elif "static" in rebal_freq_str.lower() or "inception" in rebal_freq_str.lower(): annual_mer = 0.0
            else: annual_mer = 0.0020
            
            daily_mer_fee = nav * (annual_mer / TRADING_DAYS)
            nav -= daily_mer_fee
            cash -= daily_mer_fee

            # Process Discontinuous Rebalance Executions - Cost Adjusted by live NAV
            if date in weight_map:
                target_weights = weight_map[date]
                
                # Derive current weights based on prices just before rebalancing
                actual_weights = {}
                for t, sh in shares.items():
                    p_curr = row_prices.get(t, np.nan)
                    if pd.notnull(p_curr) and not np.isnan(p_curr):
                        actual_weights[t] = (sh * p_curr) / nav if nav > 0 else 0.0
                
                all_tickers = set(actual_weights.keys()).union(target_weights.keys())
                total_weight_change = 0.0
                for t in all_tickers:
                    w_act = actual_weights.get(t, 0.0)
                    w_tgt = target_weights.get(t, 0.0)
                    total_weight_change += abs(w_tgt - w_act)
                
                # Deduct transaction cost friction directly from values (Scales with NAV)
                tx_cost = 0.0003 * total_weight_change * nav
                nav -= tx_cost
                cash -= tx_cost
                daily_tx_costs.iloc[i] = tx_cost
                daily_tx_pct.iloc[i] = 0.0003 * total_weight_change
                
                shares = {}
                long_exposure = 0.0
                short_proceeds = 0.0
                for t, w in target_weights.items():
                    if t in row_prices and pd.notnull(row_prices[t]) and row_prices[t] > 0 and w != 0:
                        shares[t] = (nav * w) / row_prices[t]
                        if w > 0:
                            long_exposure += (nav * w)
                        else:
                            short_proceeds += abs(nav * w)
                cash = nav - long_exposure + short_proceeds

            equity.iloc[i] = nav
            portfolio_cash.iloc[i] = cash

            # Record Tracking Weights
            for t, sh in shares.items():
                p_curr = row_prices.get(t, 0.0)
                if pd.notnull(p_curr) and nav > 0:
                    hist_weights.at[date, t] = (sh * p_curr) / nav
            hist_weights.at[date, "CASH"] = cash / nav if nav > 0 else 0

        # ======================================================================
        # POST-PROCESS PERFORMANCE ANALYTICS & BENCHMARK INTEGRATION
        # ======================================================================
        port_r = equity.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
        if port_r.empty or len(port_r) < 2:
            print(f"Skipping {full_index_name}: Vector holds insufficient range.")
            continue

        # Extract, align, and strictly normalize benchmark data to match starting $10,000
        bench_aligned = bench_master_series.reindex(dates).ffill().bfill()
        bench_equity = (bench_aligned / bench_aligned.iloc[0]) * START_VALUE
        bench_r = bench_equity.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)

        # Calculate excess returns using the daily RF series defined at start of loop
        excess_r = port_r - daily_rf_dec

        # Advanced Institutional Metrics Core Engine Math
        cov_matrix = np.cov(port_r, bench_r)
        covariance = cov_matrix[0, 1] if cov_matrix.ndim == 2 and cov_matrix.shape == (2, 2) else 0.0
        market_variance = cov_matrix[1, 1] if cov_matrix.ndim == 2 and cov_matrix.shape == (2, 2) else 1.0
        beta = covariance / market_variance if market_variance != 0 else 0.0

        corr_matrix = np.corrcoef(port_r, bench_r)
        correlation = corr_matrix[0, 1] if corr_matrix.ndim == 2 and corr_matrix.shape == (2, 2) else 0.0
        r_squared = correlation ** 2 if not np.isnan(correlation) else 0.0

        p_ann_mean_return = port_r.mean() * TRADING_DAYS
        bench_ann_mean_return = bench_r.mean() * TRADING_DAYS
        
        # Use the average annualized RF rate for the period
        period_rf_ann = daily_rf_series.mean()
        treynor = (p_ann_mean_return - period_rf_ann) / beta if beta != 0 else 0.0


        b_ann_vol = bench_r.std() * np.sqrt(TRADING_DAYS)
        p_sharpe = sharpe(excess_r)
        
        # Dual-Modigliani Metric Integration
        m2_absolute = period_rf_ann + (p_sharpe * b_ann_vol)
        m2_excess = m2_absolute - bench_ann_mean_return

        # Standard Risk Accounting Framework Calculations
        dd = compute_drawdown(equity)
        ttm_yield = portfolio_div_yield.rolling(window=TRADING_DAYS).sum().fillna(0)

        is_zero = dd == 0
        completed_periods = is_zero.cumsum()
        max_dd_duration = dd.groupby(completed_periods).cumcount().max()

        daily_var_95 = np.percentile(port_r, 5)
        calmar = CAGR(equity) / abs(dd.min()) if dd.min() != 0 else 0

        rolling_3m_ret = equity.pct_change(63).dropna()
        highest_3m_ret = rolling_3m_ret.max() if not rolling_3m_ret.empty else 0
        lowest_3m_ret = rolling_3m_ret.min() if not rolling_3m_ret.empty else 0

        cutoff_date = dates[-1]
        cutoff_date_str = cutoff_date.strftime('%Y-%m-%d')

        # Trailing Performance Horizons
        target_1m = cutoff_date - pd.DateOffset(months=1)
        idx_1m = equity.index[equity.index.searchsorted(target_1m)]
        ret_1m = (equity.loc[cutoff_date] / equity.loc[idx_1m]) - 1.0 if idx_1m in equity.index else np.nan

        target_3m = cutoff_date - pd.DateOffset(months=3)
        idx_3m = equity.index[equity.index.searchsorted(target_3m)]
        ret_3m = (equity.loc[cutoff_date] / equity.loc[idx_3m]) - 1.0 if idx_3m in equity.index else np.nan

        target_6m = cutoff_date - pd.DateOffset(months=6)
        idx_6m = equity.index[equity.index.searchsorted(target_6m)]
        ret_6m = (equity.loc[cutoff_date] / equity.loc[idx_6m]) - 1.0 if idx_6m in equity.index else np.nan

        ytd_boundary = pd.Timestamp(year=cutoff_date.year, month=1, day=1)
        prev_year_index = equity.index[equity.index < ytd_boundary]
        idx_ytd = prev_year_index[-1] if len(prev_year_index) > 0 else equity.index[0]
        ret_ytd = (equity.loc[cutoff_date] / equity.loc[idx_ytd]) - 1.0

        target_1y = cutoff_date - pd.DateOffset(years=1)
        valid_1y_index = equity.index[equity.index <= target_1y]
        idx_1y = valid_1y_index[-1] if len(valid_1y_index) > 0 else None
        ret_1y = (equity.loc[cutoff_date] / equity.loc[idx_1y]) - 1.0 if idx_1y and idx_1y in equity.index else np.nan

        target_3y = cutoff_date - pd.DateOffset(years=3)
        ret_3y = (equity.loc[cutoff_date] / equity.loc[equity.index[equity.index.searchsorted(target_3y)]]) ** (1 / 3) - 1.0 if equity.index[0] <= target_3y else np.nan

        target_5y = cutoff_date - pd.DateOffset(years=5)
        ret_5y = (equity.loc[cutoff_date] / equity.loc[equity.index[equity.index.searchsorted(target_5y)]]) ** (1 / 5) - 1.0 if equity.index[0] <= target_5y else np.nan

        target_10y = cutoff_date - pd.DateOffset(years=10)
        ret_10y = (equity.loc[cutoff_date] / equity.loc[equity.index[equity.index.searchsorted(target_10y)]]) ** (1 / 10) - 1.0 if equity.index[0] <= target_10y else np.nan

        ret_si = CAGR(equity)

        ann_vol = port_r.std() * np.sqrt(252)
        if ann_vol < 0.03: risk_level = "Low"
        elif ann_vol < 0.10: risk_level = "Low-Med"
        elif ann_vol < 0.18: risk_level = "Med"
        elif ann_vol < 0.30: risk_level = "Med-High"
        else: risk_level = "High"

        # Ex-Post Analytics (Clean of Framework parameters)
        stats_table = pd.DataFrame({
            "Metric": ["CAGR", "Ann. Volatility", "Sharpe Ratio", "Sortino Ratio", "Beta vs Benchmark", "R-Squared (R²)", "Treynor Ratio", "M² Absolute Measure", "M² Alpha (Excess)", "Max Drawdown", "Max DD Duration", "Daily VaR (95%)", "Calmar Ratio", "TTM Yield (End)", "Win Rate (Daily)", "Skewness", "Kurtosis"],
            "Value": [
                f"{CAGR(equity):.2%}", f"{ann_vol:.2%}", f"{sharpe(excess_r):.2f}", f"{sortino(excess_r):.2f}",
                f"{beta:.2f}", f"{r_squared:.2%}", f"{treynor:.2f}", f"{m2_absolute:.2%}", f"{m2_excess:.2%}",
                f"{dd.min():.2%}", f"{max_dd_duration} Days", f"{daily_var_95:.2%}", f"{calmar:.2f}",
                f"{ttm_yield.iloc[-1]:.2%}", f"{(port_r > 0).mean():.2%}", f"{skew(port_r):.2f}", f"{kurtosis(port_r):.2f}"
            ]
        })

        # ======================================================================
        # FUND METADATA PARSER & OVERRIDE LAYER
        # ======================================================================
        strategy_description = f"The {full_index_name} functions as a systematic rules-based factor index framework built for professional market simulation."
        base_currency = "USD"
        index_supervisor = "TBD"
        index_manager = "TBD"
        index_analysts = "TBC"

        md_path = folder / f"{name}_summary.md"
        if md_path.exists():
            try:
                md_lines = md_path.read_text(encoding="utf-8").splitlines()
                capture_desc = False
                desc_buffer = []
                for line in md_lines:
                    if "### Strategy Description" in line:
                        capture_desc = True
                        continue
                    elif line.startswith("###") or "- **Suggested" in line or "- **Index" in line:
                        capture_desc = False
                    if capture_desc:
                        desc_buffer.append(line.strip())
                    if "- **Base Currency:**" in line:
                        base_currency = line.split(":**")[-1].strip()
                    elif "- **Index Supervisor:**" in line:
                        index_supervisor = line.split(":**")[-1].strip()
                    elif "- **Index Manager:**" in line:
                        index_manager = line.split(":**")[-1].strip()
                    elif "- **Index Analysts:**" in line:
                        index_analysts = line.split(":**")[-1].strip()
                clean_desc = "<br>".join([l for l in desc_buffer if l])
                if clean_desc: strategy_description = clean_desc
            except Exception as e:
                print(f"Warning: Parser bypassing custom template layer: {e}")

        # Programmatically determine dynamic MER based on actual detected frequency matrix
        freq_lower = rebal_freq_str.lower()
        if "weekly" in freq_lower:
            suggested_mer = "0.70%"
        elif "monthly" in freq_lower:
            suggested_mer = "0.38%"
        elif "quarterly" in freq_lower:
            suggested_mer = "0.20%"
        elif "semi-annually" in freq_lower:
            suggested_mer = "0.11%"
        elif "annually" in freq_lower:
            suggested_mer = "0.06%"
        elif "static" in freq_lower or "inception" in freq_lower:
            suggested_mer = "0.00%"
        else:
            suggested_mer = "0.20%"  # Dynamic / Custom fallback baseline

        # Programmatically determine dynamic TER based ONLY on the last trailing year
        last_year_mask = dates > (cutoff_date - pd.DateOffset(years=1))
        ter_last_year_pct = daily_tx_pct[last_year_mask].sum()
        expected_ter = f"{ter_last_year_pct:.2%}"

        try:
            mer_clean = float(suggested_mer.replace("%", "").strip())
            ter_clean = float(expected_ter.replace("%", "").strip())
            expected_total_expense = f"{mer_clean + ter_clean:.2f}%"
        except Exception:
            expected_total_expense = suggested_mer

        total_tx_costs_formatted = f"${daily_tx_costs.sum():,.2f}"

        # Build clean structural framework Dataframe for multi-sheet storage
        framework_table = pd.DataFrame({
            "Parameter": ["Base Currency", "Rebalancing Cycle", "Suggested MER", "Expected TER (Last 1Y)", "Expected Total Expense Ratio", "Total Transaction Costs (SI)", "Max Revolver Usage", "Stock Borrowing Fee", "Revolver Spread", "Index Supervisor", "Index Manager", "Index Analysts"],
            "Value": [base_currency, rebal_freq_str, suggested_mer, expected_ter, expected_total_expense, total_tx_costs_formatted, f"{max_revolver_draw_pct:.2%}", f"{SHORT_BORROW_FEE:.2%}", f"{CASH_BORROW_SPREAD:.2%}", index_supervisor, index_manager, index_analysts]
        })

        # Clear Console Performance Verification Output Log
        print("\n" + "="*60)
        print(f"         PERFORMANCE ANALYTICS: {full_index_name}         ")
        print("="*60)
        print(f"CAGR                    : {CAGR(equity):.2%}")
        print(f"Annualized Volatility   : {ann_vol:.2%}")
        print(f"Sharpe Ratio            : {p_sharpe:.2f}")
        print(f"Suggested MER           : {suggested_mer}")
        print(f"Expected TER (Last 1Y)  : {expected_ter}")
        print(f"Total Transaction Costs : {total_tx_costs_formatted}")
        print("="*60 + "\n")

        stock_weights_df = hist_weights.drop("CASH", axis=1)
        active_weight_rows = hist_weights[(stock_weights_df != 0.0).any(axis=1)]
        last_active_date = active_weight_rows.index[-1] if not active_weight_rows.empty else hist_weights.index[-1]
        final_weights = hist_weights.loc[last_active_date].drop("CASH", errors="ignore")
        holdings_date_str = last_active_date.strftime('%Y-%m-%d')

        sorted_exposure_indices = final_weights.abs().sort_values(ascending=False).index
        top_10_holdings = final_weights.reindex(sorted_exposure_indices).head(10)
        top_10_holdings = top_10_holdings[top_10_holdings != 0]
        top_10_sum = top_10_holdings.sum()

        monthly_vals = equity.resample("ME").last()
        monthly_pct = monthly_vals.pct_change()
        if not monthly_vals.empty:
            monthly_pct.iloc[0] = (monthly_vals.iloc[0] / START_VALUE) - 1.0
        monthly_ret = monthly_pct.to_frame("ret")
        monthly_ret["year"] = monthly_ret.index.year.astype(str)
        monthly_ret["month"] = monthly_ret.index.month
        h_map = monthly_ret.pivot(index="year", columns="month", values="ret").reindex(columns=range(1, 13))
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        # ======================================================================
        # ARTIFACT 1: PRINT FACT SHEET PRODUCTION (PDF LAYOUT via WEASYPRINT)
        # ======================================================================
        static_chart_path = OUTPUT_DIR / f"{name}_growth_print.png"
        fig_mpl, ax_mpl = plt.subplots(figsize=(7.5, 1.25)) 
        ax_mpl.plot(equity.index, equity.values, color=MAIN_COLOR, linewidth=1.5, label="Strategy Portfolio")
        ax_mpl.plot(bench_equity.index, bench_equity.values, color="#7A8B99", linewidth=1.1, linestyle="--", label="Russell 3000 Benchmark")
        ax_mpl.set_facecolor('white')
        ax_mpl.grid(True, linestyle='--', alpha=0.3, color='#CBD5E0')
        ax_mpl.tick_params(colors='#4A5568', labelsize=7)
        ax_mpl.set_ylabel("NAV ($)", fontsize=7.5, color='#4A5568', fontweight='bold')
        ax_mpl.legend(loc="upper left", fontsize=6, frameon=True, facecolor="white", edgecolor="none")
        for spine in ['top', 'right', 'left', 'bottom']:
            ax_mpl.spines[spine].set_color('#E2E8F0')
        plt.tight_layout()
        plt.savefig(static_chart_path, dpi=300, bbox_inches='tight')
        plt.close()

        chart_uri = static_chart_path.resolve().as_uri()
        logo_uri = LOGO_PATH.resolve().as_uri() if LOGO_PATH.exists() else ""
        theta_logo_uri = THETA_LOGO_PATH.resolve().as_uri() if THETA_LOGO_PATH.exists() else ""

        sponsor_html_elements = []
        assets_dir = Path("jmtl_assets")
        if assets_dir.exists():
            for img_file in assets_dir.iterdir():
                if img_file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    if img_file.name not in ["JMTL Logo.jpg", "Theta Data Logo.png"]:
                        s_uri = img_file.resolve().as_uri()
                        sponsor_html_elements.append(f'<img src="{s_uri}" class="sponsor-logo">')
        sponsors_logos_html = "".join(sponsor_html_elements)

        letter_css = f"""
        @page {{
            size: letter portrait;
            margin: 4mm 8mm 4mm 8mm;
            @bottom-right {{
                content: "Official Asgard Index Fact Sheet • Page 1 of 1";
                font-family: Arial, sans-serif;
                font-size: 5.5pt;
                color: #A3A3A3;
            }}
        }}
        body {{ font-family: Arial, sans-serif; color: #2D3748; line-height: 1.25; margin: 0; padding: 0; }}
        .header-table {{ width: 100%; table-layout: fixed; border-bottom: 2px solid {MAIN_COLOR}; padding-bottom: 2px; margin-bottom: 4px; }}
        .index-title {{ font-size: 14pt; font-weight: bold; color: {MAIN_COLOR}; letter-spacing: -0.5px; }}
        .meta-box {{ font-size: 6.5pt; color: #4A5568; text-align: right; white-space: nowrap; }}
        .section-title {{ font-size: 7.5pt; font-weight: bold; color: #FFFFFF; background-color: {MAIN_COLOR}; padding: 2px 4px; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 3px; margin-bottom: 2px; }}
        
        .outer-split-table {{ width: 100%; table-layout: fixed; border-collapse: collapse; }}
        .stats-table, .holdings-table, .heatmap-table, .trailing-table {{ width: 100%; table-layout: fixed; border-collapse: collapse; }}
        
        .stats-table td, .holdings-table td {{ 
            padding: 0.8px 3px; 
            font-size: 5.8pt; 
            border-bottom: 1px solid #E2E8F0; 
            white-space: normal; 
            word-wrap: break-word;
            vertical-align: middle;
        }}
        .stats-table tr:nth-child(even), .holdings-table tr:nth-child(even) {{ background-color: #F8FAFC; }}
        .stats-label {{ color: #4A5568; font-weight: 500; }}
        .stats-val {{ text-align: right; font-weight: bold; color: {MAIN_COLOR}; }}
        
        .trailing-table th {{ font-size: 5.5pt; background-color: #EDF2F7; font-weight: bold; padding: 3px 2px; border: 1px solid #CBD5E0; text-align: center; }}
        .trailing-table td {{ font-size: 6.2pt; padding: 3px 2px; border: 1px solid #E2E8F0; text-align: center; background-color: #FFFFFF; }}
        .heatmap-table th {{ font-size: 5.8pt; background-color: #EDF2F7; font-weight: bold; padding: 2px 1px; border: 1px solid #CBD5E0; }}
        .heatmap-table td {{ font-size: 5.8pt; padding: 2px 1px; border: 1px solid #CBD5E0; font-weight: 500; text-align: center; }}
        .desc-text {{ font-size: 6.2pt; color: #4A5568; text-align: justify; line-height: 1.35; }}
        .chart-box {{ text-align: center; margin: 0; }}
        .chart-img {{ width: 100%; height: auto; }}
        
        .footer-line-container {{ margin-top: 6px; }}
        .footer-delimiter {{ border-top: 1.5px solid {MAIN_COLOR}; margin: 2px 0; }}
        .footer-sub-delimiter {{ border-top: 1px solid #CBD5E0; margin: 2px 0; }}
        
        .footer-block {{ width: 100%; text-align: center; padding: 8px 0; }}
        .footer-title {{ font-size: 8.5pt; font-weight: bold; color: {MAIN_COLOR}; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; text-align: center; }}
        .logo-center-container {{ text-align: center; display: block; width: 100%; }}
        .thetadata-logo {{ height: 38px; width: auto; display: inline-block; margin: 0 auto; }}
        .sponsor-logo {{ height: 35px; width: auto; margin: 0 16px; display: inline-block; vertical-align: middle; }}
        .no-logo-fallback {{ color: #718096; font-style: italic; font-size: 7.5pt; text-transform: none; display: inline-block; }}
        
        .disclaimer-box {{ margin-top: 4px; padding-top: 2px; border-top: 1px solid #CBD5E0; }}
        .disclaimer-title {{ font-size: 5.5pt; font-weight: bold; color: #1A202C; text-transform: uppercase; margin-bottom: 1px; }}
        .disclaimer-text {{ font-size: 5.1pt; color: #718096; text-align: justify; line-height: 1.15; }}
        
        .risk-table {{ width: 100%; table-layout: fixed; border-collapse: collapse; margin-top: 4px; }}
        .risk-cell {{ font-size: 5.2pt; font-weight: bold; text-align: center; border: 1px solid #CBD5E0; padding: 2.5px 0; color: #A0AEC0; background-color: #FFFFFF; text-transform: uppercase; }}
        .risk-active {{ background-color: {MAIN_COLOR}; color: #FFFFFF; border-color: {MAIN_COLOR}; }}
        """

        full_stats_rows = "".join(f'<tr><td class="stats-label">{row["Metric"]}</td><td class="stats-val">{row["Value"]}</td></tr>' for _, row in stats_table.iterrows())

        holdings_rows = "".join(f'<tr><td class="stats-label">{ticker}</td><td class="stats-val">{weight:.2%}</td></tr>' for ticker, weight in top_10_holdings.items())
        holdings_rows += f'<tr style="background-color: #EDF2F7; font-weight: bold;"><td class="stats-label" style="border-top: 1px solid #CBD5E0;"><strong>Net Top 10 Allocation</strong></td><td class="stats-val" style="border-top: 1px solid #CBD5E0; color:{MAIN_COLOR};"><strong>{top_10_sum:.2%}</strong></td></tr>'

        pdf_heatmap_rows = ""
        for yr in h_map.index:
            row_str = f"<tr><td style='font-weight:bold; background-color:#EDF2F7;'>{yr}</td>"
            for m in range(1, 13):
                val = h_map.loc[yr, m]
                if pd.isnull(val):
                    row_str += "<td style='background-color: #FFFFFF; color: #CBD5E0;'>-</td>"
                else:
                    bg = "#DEF7EC" if val > 0 else "#FDE8E8"
                    color = "#03543F" if val > 0 else "#9B1C1C"
                    row_str += f"<td style='background-color: {bg}; color: {color};'>{val:.1%}</td>"
            row_str += "</tr>"
            pdf_heatmap_rows += row_str

        def format_horizon_val(val):
            if pd.isnull(val) or np.isnan(val): return "<td>-</td>"
            return f'<td style="font-weight: bold; color: #1A202C;">{val:.2%}</td>'

        logo_html = f'<img src="{logo_uri}" style="max-height: 22px;">' if logo_uri else f'<span style="font-size:9pt; font-weight:bold; color:{MAIN_COLOR}">Asgard</span>'

        factsheet_html = f"""
        <!DOCTYPE html>
        <html>
        <head><meta charset="utf-8"></head>
        <body>
            <table class="header-table">
                <colgroup><col style="width: 55%;"><col style="width: 45%;"></colgroup>
                <tr>
                    <td style="vertical-align: middle;"><div class="index-title">{full_index_name} INDEX</div></td>
                    <td style="vertical-align: middle; text-align: right;">
                        {logo_html}
                        <div class="meta-box">
                            <strong>Institutional Performance Profile</strong><br>
                            Base Currency: {base_currency} &nbsp;|&nbsp; Inception: {inception_date_str} &nbsp;|&nbsp; Cutoff: {cutoff_date_str}
                        </div>
                    </td>
                </tr>
            </table>
            <div class="section-title">Index Architecture & Fund Parameters</div>
            <table class="outer-split-table" style="margin-bottom: 2px;">
                <colgroup><col style="width: 40%;"><col style="width: 30%;"><col style="width: 30%;"></colgroup>
                <tr>
                    <td style="vertical-align: top; padding-right: 14px;">
                        <div style="font-size: 6.5pt; font-weight: bold; color: {MAIN_COLOR}; margin-bottom: 2px;">Index Strategy Description</div>
                        <div class="desc-text">{strategy_description}</div>
                    </td>
                    
                    <td style="vertical-align: top; padding-right: 14px;">
                        <div style="font-size: 6.5pt; font-weight: bold; color: {MAIN_COLOR}; margin-bottom: 2px;">Fund Framework Parameters</div>
                                 <table class="stats-table" style="border: 1px solid #E2E8F0;">
                                     <tr><td class="stats-label">Rebalancing Cycle</td><td class="stats-val">{rebal_freq_str}</td></tr>
                                     <tr><td class="stats-label">Suggested MER</td><td class="stats-val">{suggested_mer}</td></tr>
                                     <tr><td class="stats-label">Expected TER (Last 1Y)</td><td class="stats-val">{expected_ter}</td></tr>
                                     <tr><td class="stats-label">Expected Total Expense Ratio</td><td class="stats-val">{expected_total_expense}</td></tr>
                                     <tr style="background-color: #EDF2F7; font-weight: bold;"><td class="stats-label">Total Transaction Costs (SI)</td><td class="stats-val" style="color: {MAIN_COLOR};">{total_tx_costs_formatted}</td></tr>
                                     <tr><td class="stats-label">Max Revolver Usage</td><td class="stats-val">{max_revolver_draw_pct:.2%}</td></tr>
                                     <tr><td class="stats-label">Stock Borrowing Fee</td><td class="stats-val">{SHORT_BORROW_FEE:.2%}</td></tr>
                                     <tr><td class="stats-label">Revolver Spread</td><td class="stats-val">{CASH_BORROW_SPREAD:.2%}</td></tr>
                                     <tr><td class="stats-label">Index Supervisor</td><td class="stats-val">{index_supervisor}</td></tr>
                                     <tr><td class="stats-label">Index Manager</td><td class="stats-val">{index_manager}</td></tr>
                                     <tr><td class="stats-label">Index Analysts</td><td class="stats-val">{index_analysts}</td></tr>
                                 </table>

                        <div style="font-size: 6.5pt; font-weight: bold; color: {MAIN_COLOR}; margin-top: 4px; margin-bottom: 1px;">Volatility Risk Scale</div>
                        <table class="risk-table">
                            <tr>
                                <td class="risk-cell {'risk-active' if risk_level == 'Low' else ''}">Low</td>
                                <td class="risk-cell {'risk-active' if risk_level == 'Low-Med' else ''}">L-M</td>
                                <td class="risk-cell {'risk-active' if risk_level == 'Med' else ''}">Med</td>
                                <td class="risk-cell {'risk-active' if risk_level == 'Med-High' else ''}">M-H</td>
                                <td class="risk-cell {'risk-active' if risk_level == 'High' else ''}">High</td>
                            </tr>
                        </table>
                    </td>
                    
                    <td style="vertical-align: top;">
                        <div style="font-size: 6.5pt; font-weight: bold; color: {MAIN_COLOR}; margin-bottom: 2px;">Ex-Post Quantitative Analytics</div>
                        <table class="stats-table" style="border: 1px solid #E2E8F0;">
                            {full_stats_rows}
                        </table>
                    </td>
                </tr>
            </table>
            <div class="section-title">Growth of $10,000 Capital Trajectory vs Benchmark (After Estimated Trading Costs)</div>
            <div class="chart-box"><img class="chart-img" src="{chart_uri}"></div>
            <div class="section-title">Trailing Performance Horizons</div>
            <table class="trailing-table" style="margin-bottom: 3px;">
                <thead><tr><th>1M</th><th>3M</th><th>6M</th><th>YTD</th><th>1Y</th><th>3Y (Ann.)</th><th>5Y (Ann.)</th><th>10Y (Ann.)</th><th>SI (Ann.)</th></tr></thead>
                <tbody><tr>
                    {format_horizon_val(ret_1m)}{format_horizon_val(ret_3m)}{format_horizon_val(ret_6m)}{format_horizon_val(ret_ytd)}
                    {format_horizon_val(ret_1y)}{format_horizon_val(ret_3y)}{format_horizon_val(ret_5y)}{format_horizon_val(ret_10y)}{format_horizon_val(ret_si)}
                </tr></tbody>
            </table>
            <table class="outer-split-table">
                <colgroup><col style="width: 65%;"><col style="width: 35%;"></colgroup>
                <tr>
                    <td style="vertical-align: top; padding-right: 10px;">
                        <div class="section-title">Historical Monthly Return Matrix</div>
                        <table class="heatmap-table">
                            <thead><tr><th>Year</th>{"".join(f'<th>{m}</th>' for m in month_names)}</tr></thead>
                            <tbody>{pdf_heatmap_rows}</tbody>
                        </table>
                    </td>
                    <td style="vertical-align: top;">
                        <div class="section-title">Top 10 Largest Exposures ({holdings_date_str})</div>
                        <table class="holdings-table">
                            <thead><tr style="background-color: #EDF2F7; font-weight: bold; font-size: 6pt;"><td style="padding: 2px; border: 1px solid #CBD5E0;">Ticker</td><td style="padding: 2px; border: 1px solid #CBD5E0; text-align: right;">Weight</td></tr></thead>
                            <tbody>{holdings_rows}</tbody>
                        </table>
                    </td>
                </tr>
            </table>

            <div class="footer-line-container">
                <div class="footer-delimiter"></div>
                <div class="footer-block">
                    <div class="footer-title">Powered By Theta Data</div>
                    <div class="logo-center-container">
                        {f'<img src="{theta_logo_uri}" class="thetadata-logo">' if theta_logo_uri else '<span class="no-logo-fallback">[Data Infrastructure Core]</span>'}
                    </div>
                </div>
                <div class="footer-sub-delimiter"></div>
                <div class="footer-block">
                    <div class="footer-title">Corporate & Academic Sponsors</div>
                    <div class="logo-center-container">
                        {sponsors_logos_html if sponsors_logos_html else '<span class="no-logo-fallback">Academic Infrastructure Partners</span>'}
                    </div>
                </div>
            </div>

            <div class="disclaimer-box">
                <div class="disclaimer-title">Quebec Jurisdictional Disclosures & Student-Run Project Disclaimer</div>
                <div class="disclaimer-text">
                    <strong>Educational Initiative Notice:</strong> This profile document is compiled completely automatically by a locked analytical database framework. This benchmark strategy and all collateral performance metrics are engineered strictly as an academic, student-run research project. This material is designed solely for professional simulation, competitive training, and internal institutional reference criteria. It does not constitute professional investment, legal, or tax advice, nor does it represent an express or implied recommendation, offer, or commercial solicitation to allocate live capital, trade securities, or deploy investment assets in any active portfolio strategy. 
                    <br><br>
                    <strong>Regulatory Compliance Statement:</strong> The index calculation engine, the backtested data layer, and the operators of Project Asgard are not registered with or recognized by the <i>Autorité des marchés financiers (AMF)</i> of Quebec, the Canadian Securities Administrators (CSA), or any other regulatory authority in Canada or international jurisdictions. Transactions and data represented here do not correspond to live market execution records, real exchange operations, or registered investment funds. Backtested performance curves are entirely simulated and theoretical; they do not account for physical portfolio execution slippage, broker commission metrics, cash borrow constraints, transactional friction, or localized fiscal and liquidating tax liabilities within the Province of Quebec or federal frameworks. <strong>Past performance figures provide no reliable indicator, benchmark baseline, or guarantee of future real-world capital returns.</strong>
                    <br><br>
                    All investments naturally carry severe market risks, up to and including the permanent, irrecoverable loss of all deployed principal capital. Under a theoretical historical baseline modeling approach, a fixed initial capital commitment of $10,000.00 deployed directly into this strategy at the sample inception of {inception_date_str} would have evolved to a total closing portfolio value of <strong>${equity.iloc[-1]:,.2f}</strong> as of the current reporting date ({cutoff_date_str}). Under observation of trailing sub-windows within this historical range, an asset manager executing this logic would have experienced a maximum rolling 3-month performance gain of <strong>{highest_3m_ret:.2%}</strong>, while the corresponding worst historical rolling 3-month trailing window would have triggered an absolute capital drawdown loss of <strong>{lowest_3m_ret:.2%}</strong>. The index compilation framework and its operators disclaim any explicit or implicit liability for localized data gaps, processing variances, or decisions enacted using this system.
                </div>
            </div>
        </body>
        </html>
        """
        HTML(string=factsheet_html).write_pdf(OUTPUT_DIR / f"{name}_factsheet.pdf", stylesheets=[CSS(string=letter_css)])

        # ======================================================================
        # ARTIFACT 2: INTERACTIVE DASHBOARD PRODUCTION (HTML INTERACTIVE REPORT)
        # ======================================================================
        fig_plotly = make_subplots(
            rows=6, cols=1,
            subplot_titles=("Equity Curve vs Russell 3000 Benchmark ($10K Normalized)", "Drawdown Profile", "Rolling Risk Coefficients (252D)", "TTM Dividend Yield Trajectory", "Ex-Post Daily Return Distribution", "Monthly Returns Performance Matrix"),
            vertical_spacing=0.06
        )
        fig_plotly.add_trace(go.Scatter(x=equity.index, y=equity, name="Strategy NAV", line=dict(color='royalblue', width=2)), row=1, col=1)
        fig_plotly.add_trace(go.Scatter(x=bench_equity.index, y=bench_equity, name="Russell 3000 Benchmark", line=dict(color='slategrey', width=1.5, dash='dash')), row=1, col=1)
        
        fig_plotly.add_trace(go.Scatter(x=dd.index, y=dd, name="Drawdown", fill='tozeroy', line=dict(color='firebrick', width=1.5)), row=2, col=1)
        fig_plotly.add_trace(go.Scatter(x=port_r.index, y=rolling_metric(port_r, sharpe), name="Rolling Sharpe", line=dict(color='mediumseagreen')), row=3, col=1)
        fig_plotly.add_trace(go.Scatter(x=port_r.index, y=rolling_metric(port_r, sortino), name="Rolling Sortino", line=dict(color='darkorange')), row=3, col=1)
        fig_plotly.add_trace(go.Scatter(x=ttm_yield.index, y=ttm_yield, name="TTM Yield", line=dict(color='gold', width=2)), row=4, col=1)
        fig_plotly.add_trace(go.Histogram(x=port_r, name="Daily Returns Density", marker_color='slategrey', opacity=0.85), row=5, col=1)
         
        z_text = [[f"{val:.1%}" if pd.notnull(val) else "" for val in row] for row in h_map.values]
        fig_plotly.add_trace(go.Heatmap(
            z=h_map.values, x=month_names, y=h_map.index,
            colorscale="RdYlGn", zmid=0, showscale=False,
            text=z_text, texttemplate="%{text}", textfont={"size": 11, "color": "black"},
            xgap=1.5, ygap=1.5
        ), row=6, col=1)

        fig_plotly.update_yaxes(type='category', row=6, col=1)
        fig_plotly.update_layout(height=2300, title_text=f"Performance Analytics Report Dashboard: {full_index_name}", showlegend=True, template="plotly_white")

        stats_table_html_clean = stats_table.copy()
        summary_html_snippet = stats_table_html_clean.to_html(index=False, classes='table table-sm table-hover table-striped table-bordered text-center').replace('border="1"', 'border="0"')
        
        full_html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{full_index_name} Performance Report</title>
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
            <style>
                body {{ background-color: #f8f9fa; font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; padding: 40px 15px; }}
                .report-card {{ background: #ffffff; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); padding: 30px; margin-bottom: 30px; }}
                .table-wrapper {{ border-radius: 6px; overflow: hidden; }}
                thead th {{ background-color: #343a40 !important; color: #ffffff !important; font-weight: 600; }}
                h2 {{ color: #212529; font-weight: 700; letter-spacing: -0.5px; }}
            </style>
        </head>
        <body>
            <div class="container-fluid max-width: 1400px;">
                <div class="report-card">
                    <h2 class="mb-4 text-dark">{full_index_name} Index Strategy Statistics Summary</h2>
                    <div class="row">
                        <div class="col-xl-4 col-lg-5 col-md-12 mb-4">
                            <div class="p-3 mb-3 bg-white rounded shadow-sm border">
                                <h6 class="font-weight-bold text-uppercase text-muted mb-2" style="font-size: 0.75rem;">Fund Framework Parameters</h6>
                                 <table class="table table-sm style='font-size:0.75rem;' mb-0">
                                     <tr><td><strong>Base Currency</strong></td><td>{base_currency}</td></tr>
                                     <tr><td><strong>Rebal Cycle</strong></td><td>{rebal_freq_str}</td></tr>
                                     <tr><td><strong>Suggested MER</strong></td><td>{suggested_mer}</td></tr>
                                     <tr><td><strong>Expected TER (Last 1Y)</strong></td><td>{expected_ter}</td></tr>
                                     <tr><td><strong>Expected Total Expense Ratio</strong></td><td>{expected_total_expense}</td></tr>
                                     <tr class="table-info"><td><strong>Total Transaction Costs (SI)</strong></td><td><strong>{total_tx_costs_formatted}</strong></td></tr>
                                     <tr><td><strong>Max Revolver Usage</strong></td><td>{max_revolver_draw_pct:.2%}</td></tr>
                                     <tr><td><strong>Stock Borrowing Fee</strong></td><td>{SHORT_BORROW_FEE:.2%}</td></tr>
                                     <tr><td><strong>Revolver Spread</strong></td><td>{CASH_BORROW_SPREAD:.2%}</td></tr>
                                     <tr><td><strong>Index Supervisor</strong></td><td>{index_supervisor}</td></tr>
                                     <tr><td><strong>Index Manager</strong></td><td>{index_manager}</td></tr>
                                     <tr><td><strong>Index Analysts</strong></td><td>{index_analysts}</td></tr>
                                 </table>
                            </div>
                            <div class="p-3 mb-3 bg-white rounded shadow-sm border">
                                <h6 class="font-weight-bold text-uppercase text-muted mb-2" style="font-size: 0.75rem;">Index Volatility Risk Profile</h6>
                                <div class="d-flex text-center font-weight-bold" style="font-size: 0.72rem;">
                                    <div class="p-2 border flex-fill {'text-white' if risk_level=='Low' else 'text-muted'}" style="background-color: {'#0A192F' if risk_level=='Low' else '#ffffff'};">Low</div>
                                    <div class="p-2 border flex-fill {'text-white' if risk_level=='Low-Med' else 'text-muted'}" style="background-color: {'#0A192F' if risk_level=='Low-Med' else '#ffffff'};">Low-Med</div>
                                    <div class="p-2 border flex-fill {'text-white' if risk_level=='Med' else 'text-muted'}" style="background-color: {'#0A192F' if risk_level=='Med' else '#ffffff'};">Med</div>
                                    <div class="p-2 border flex-fill {'text-white' if risk_level=='Med-High' else 'text-muted'}" style="background-color: {'#0A192F' if risk_level=='Med-High' else '#ffffff'};">Med-High</div>
                                    <div class="p-2 border flex-fill {'text-white' if risk_level=='High' else 'text-muted'}" style="background-color: {'#0A192F' if risk_level=='High' else '#ffffff'};">High</div>
                                </div>
                            </div>
                            <div class="table-wrapper shadow-sm">{summary_html_snippet}</div>
                        </div>
                    </div>
                    <div class="row">
                        <div class="col-12 mt-2">
                            <div class="p-2 border rounded bg-light">{fig_plotly.to_html(full_html=False, include_plotlyjs='cdn')}</div>
                        </div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        (OUTPUT_DIR / f"{name}_report.html").write_text(full_html_report, encoding="utf-8")

        # ======================================================================
        # ARTIFACT 3: TRANSACTIONAL DATA BASELINE WORKBOOK (MULTI-SHEET EXCEL)
        # ======================================================================
        daily_asset_returns = prices.pct_change().fillna(0)
        daily_metrics_log = pd.DataFrame({
            "Equity_NAV": equity,
            "Benchmark_NAV": bench_equity,
            "Absolute_Cash": portfolio_cash,
            "Drawdown": dd,
            "TTM_Dividend_Yield": ttm_yield,
            "Portfolio_Daily_Return": port_r,
            "Benchmark_Daily_Return": bench_r,
            "Transaction_Costs_Incurred": daily_tx_costs
        }, index=dates).fillna(0)

        with pd.ExcelWriter(OUTPUT_DIR / f"{name}_details.xlsx", engine="openpyxl") as writer:
            framework_table.to_excel(writer, sheet_name="Fund_Parameters", index=False)
            stats_table_html_clean.to_excel(writer, sheet_name="ExPost_Analytics", index=False)
            daily_metrics_log.to_excel(writer, sheet_name="Daily_Portfolio_Data")
            hist_weights.to_excel(writer, sheet_name="Historical_Asset_Weights")
            daily_asset_returns.to_excel(writer, sheet_name="Underlying_Asset_Returns")
            hist_contribs.to_excel(writer, sheet_name="Daily_Return_Contributions")
            hist_contribs.resample("ME").sum().to_excel(writer, sheet_name="Monthly_Attribution_Summary")

        print(f"Successfully Finalized Core Artifact Pipeline Maps For: {full_index_name}")