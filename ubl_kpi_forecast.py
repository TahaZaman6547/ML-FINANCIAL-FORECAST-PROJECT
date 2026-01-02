import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from matplotlib.backends.backend_pdf import PdfPages


# =========================================
# CONFIG
# =========================================

DATA_PATH = "Data/UBL_Financials_.xlsx"   # Excel file path
DATE_COL = "Date"

TEST_STEPS = 4          # last 4 quarters = backtest
FUTURE_YEARS = 5        # 2026–2030
FUTURE_STEPS = FUTURE_YEARS * 4
N_LAGS = 4              # number of lags in Ridge features

# Excel column names for KPIs
KPI_COLS = {
    "Deposits": "Deposits and other accounts",
    "Advances": "Advances",
    "Investments": "Investments",
    "Borrowings": "Borrowings",
    "Total_Assets": "TOTAL ASSETS",
    "Total_Equity": "TOTAL EQUITY",
    "Unappropriated_Profit": "Unappropriated profit",
}


# =========================================
# 1. LOAD & PREPARE DATA
# =========================================

def load_data() -> pd.DataFrame:
    """Load UBL_Financials_.xlsx and return cleaned numeric DataFrame indexed by Date."""
    df = pd.read_excel(DATA_PATH)

    # normalize column names
    df.columns = [str(c).strip() for c in df.columns]

    # date handling
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL)

    # remove duplicates if any
    if df[DATE_COL].duplicated().any():
        print("⚠ Duplicate dates detected – keeping last row per date")
        df = df.drop_duplicates(subset=[DATE_COL], keep="last")

    df = df.set_index(DATE_COL)

    # numeric columns only
    df = df.select_dtypes(include=["number"])

    # forward fill gaps
    df = df.ffill()

    # check KPI columns exist
    missing = [v for v in KPI_COLS.values() if v not in df.columns]
    if missing:
        raise ValueError(f"Missing KPI columns in Excel: {missing}")

    print("Data loaded:", df.shape)
    print("Date range:", df.index.min().date(), "→", df.index.max().date())
    return df


# =========================================
# 2. BUILD KPI FRAME + RATIOS
# =========================================

def build_kpi_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build compact KPI panel:
    - Levels (Deposits, Advances, etc.)
    - Ratios including ROE (using change in Unappropriated profit)
    """
    k = pd.DataFrame(index=df.index)

    k["Deposits"] = df[KPI_COLS["Deposits"]]
    k["Advances"] = df[KPI_COLS["Advances"]]
    k["Investments"] = df[KPI_COLS["Investments"]]
    k["Borrowings"] = df[KPI_COLS["Borrowings"]]
    k["Total_Assets"] = df[KPI_COLS["Total_Assets"]]
    k["Total_Equity"] = df[KPI_COLS["Total_Equity"]]
    k["Unappropriated_Profit"] = df[KPI_COLS["Unappropriated_Profit"]]

    # PAT proxy ~ change in Unappropriated profit
    pat_proxy = k["Unappropriated_Profit"].diff()

    avg_equity = k["Total_Equity"].rolling(2).mean()
    roe_quarterly = pat_proxy / avg_equity
    k["ROE_Annualised_pct"] = (roe_quarterly * 4.0) * 100.0

    k["LDR"] = k["Advances"] / k["Deposits"]          # Loan / Deposit
    k["Equity_to_Assets"] = k["Total_Equity"] / k["Total_Assets"]

    return k


# =========================================
# 3. FEATURE ENGINEERING & MODELLING
# =========================================

def make_lag_features(series: pd.Series, n_lags: int = N_LAGS):
    """Create supervised lag features for a 1D series."""
    data = pd.DataFrame({"y": series})
    for lag in range(1, n_lags + 1):
        data[f"lag{lag}"] = series.shift(lag)

    data = data.dropna()

    X = data[[f"lag{lag}" for lag in range(1, n_lags + 1)]]
    y = data["y"]

    # train/test split based on last TEST_STEPS observations
    if len(y) <= TEST_STEPS:
        raise ValueError("Not enough data after lags to create train/test split.")

    X_train = X.iloc[:-TEST_STEPS]
    X_test = X.iloc[-TEST_STEPS:]
    y_train = y.iloc[:-TEST_STEPS]
    y_test = y.iloc[-TEST_STEPS:]

    return X_train, X_test, y_train, y_test


def train_and_backtest(series: pd.Series, name: str):
    """
    Train Naive, SARIMAX, Ridge on a single KPI and backtest on last TEST_STEPS.
    Returns:
        backtest_df, metrics_df, models_dict, best_model_name
    """
    X_train, X_test, y_train, y_test = make_lag_features(series)

    # Force test window to exactly TEST_STEPS
    y_test = y_test[-TEST_STEPS:]
    X_test = X_test[-TEST_STEPS:]

    # ===== Naive (random walk) =====
    naive_pred = np.repeat(y_train.iloc[-1], TEST_STEPS)

    # ===== SARIMAX =====
    sarimax = SARIMAX(
        y_train,
        order=(1, 1, 1),
        seasonal_order=(0, 1, 1, 4),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    sar_pred = sarimax.get_forecast(TEST_STEPS).predicted_mean.values

    # ===== Ridge on lags =====
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    rg_pred = ridge.predict(X_test)

    # ===== Metrics =====
    def mape(a, p) -> float:
        a = np.asarray(a, dtype="float64")
        p = np.asarray(p, dtype="float64")
        mask = a != 0
        if mask.sum() == 0:
            return np.nan
        return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100.0)

    metrics = pd.DataFrame(
        {
            "KPI": [name] * 3,
            "Model": ["SARIMAX", "Ridge", "Naive"],
            "MAE": [
                mean_absolute_error(y_test, sar_pred),
                mean_absolute_error(y_test, rg_pred),
                mean_absolute_error(y_test, naive_pred),
            ],
            "RMSE": [
                np.sqrt(mean_squared_error(y_test, sar_pred)),
                np.sqrt(mean_squared_error(y_test, rg_pred)),
                np.sqrt(mean_squared_error(y_test, naive_pred)),
            ],
            "MAPE": [
                mape(y_test, sar_pred),
                mape(y_test, rg_pred),
                mape(y_test, naive_pred),
            ],
        }
    )

    # Prefer SARIMAX/Ridge over Naive if difference small
    metrics_sorted = metrics.sort_values("MAPE")
    best_model = metrics_sorted.iloc[0]["Model"]
    if best_model == "Naive" and len(metrics_sorted) > 1:
        if metrics_sorted.iloc[1]["MAPE"] <= metrics_sorted.iloc[0]["MAPE"] + 1.0:
            best_model = metrics_sorted.iloc[1]["Model"]

    models = {"SARIMAX": sarimax, "Ridge": ridge}

    # Backtest dataframe (aligned on y_test index)
    backtest = pd.DataFrame(
        {
            "Actual": y_test.values,
            "SARIMAX": sar_pred,
            "Ridge": rg_pred,
            "Naive": naive_pred,
        },
        index=y_test.index,
    )

    return backtest, metrics, models, best_model


def forecast_future(series: pd.Series, models: dict, best_model: str) -> pd.DataFrame:
    """Produce FUTURE_STEPS quarterly forecasts and return a DF with Naive/SARIMAX/Ridge/Best."""
    last_date = series.index[-1]
    future_index = pd.date_range(
        last_date + pd.offsets.QuarterEnd(), periods=FUTURE_STEPS, freq="Q"
    )

    # Naive
    naive_future = np.repeat(series.iloc[-1], FUTURE_STEPS)

    # SARIMAX
    sarimax = models["SARIMAX"]
    sar_future = sarimax.get_forecast(steps=FUTURE_STEPS).predicted_mean.values

    # Ridge (iterative)
    ridge = models["Ridge"]
    hist_vals = series.dropna().values.tolist()
    ridge_vals = []
    for _ in range(FUTURE_STEPS):
        lag_dict = {f"lag{lag}": hist_vals[-lag] for lag in range(1, N_LAGS + 1)}
        x_new = pd.DataFrame([lag_dict])
        y_hat = ridge.predict(x_new)[0]
        ridge_vals.append(y_hat)
        hist_vals.append(y_hat)
    ridge_future = np.array(ridge_vals)

    df_future = pd.DataFrame(
        {
            "Naive": naive_future,
            "SARIMAX": sar_future,
            "Ridge": ridge_future,
        },
        index=future_index,
    )

    df_future["Best"] = df_future[best_model]
    return df_future


# =========================================
# 4. MAIN PIPELINE
# =========================================

def main():

    raw = load_data()
    kpi_hist = build_kpi_frame(raw)

    level_kpis = [
        "Deposits",
        "Advances",
        "Investments",
        "Borrowings",
        "Total_Assets",
        "Total_Equity",
        "Unappropriated_Profit",
    ]

    all_metrics = []
    backtests = {}
    future_levels = {}

    # -------------------------------
    # Run models per KPI
    # -------------------------------
    for kpi in level_kpis:
        print(f"\nProcessing {kpi}")
        series = kpi_hist[kpi].dropna()

        if len(series) <= (TEST_STEPS + N_LAGS):
            print(f"   Not enough history for {kpi}, skipping.")
            continue

        bt, metrics, models, best = train_and_backtest(series, kpi)
        fc_future = forecast_future(series, models, best)

        metrics["Best_Model"] = best
        backtests[kpi] = bt
        future_levels[kpi] = fc_future["Best"]
        all_metrics.append(metrics)

    metrics_all = pd.concat(all_metrics, ignore_index=True)
    future_levels_df = pd.DataFrame(future_levels)

    # -------------------------------
    # Future ratios from forecasted levels
    # -------------------------------
    ratios_future = pd.DataFrame(index=future_levels_df.index)
    ratios_future["LDR"] = future_levels_df["Advances"] / future_levels_df["Deposits"]
    ratios_future["Equity_to_Assets"] = (
        future_levels_df["Total_Equity"] / future_levels_df["Total_Assets"]
    )

    up = future_levels_df["Unappropriated_Profit"]
    pat_proxy_f = up.diff()
    avg_eq_f = future_levels_df["Total_Equity"].rolling(2).mean()
    roe_q_f = pat_proxy_f / avg_eq_f
    ratios_future["ROE_Annualised_pct"] = (roe_q_f * 4.0) * 100.0

    # -------------------------------
    # Yearly management summary
    # -------------------------------
    combined_levels = pd.concat([kpi_hist[level_kpis], future_levels_df], axis=0)
    yearly_levels = combined_levels.resample("A-DEC").last()

    ratios_combined = pd.concat(
        [kpi_hist[["LDR", "Equity_to_Assets", "ROE_Annualised_pct"]], ratios_future],
        axis=0,
    )
    yearly_ratios = ratios_combined.resample("A-DEC").mean()

    yearly_summary = pd.concat([yearly_levels, yearly_ratios], axis=1)

    # =================================
    # 5. SAVE EXCEL
    # =================================
    with pd.ExcelWriter("UBL_KPI_Forecast_Output.xlsx") as writer:
        kpi_hist.to_excel(writer, sheet_name="Historical_KPIs")
        future_levels_df.to_excel(writer, sheet_name="Future_KPIs_Quarterly")
        yearly_summary.to_excel(writer, sheet_name="Yearly_Summary")
        metrics_all.to_excel(writer, sheet_name="Model_Performance", index=False)

        for kpi, bt in backtests.items():
            sheet_name = f"Backtest_{kpi[:25]}"
            bt.to_excel(writer, sheet_name=sheet_name)

    print("\n📊 Excel saved as UBL_KPI_Forecast_Output.xlsx")

    # =================================
    # 6. SAVE PDF REPORT
    # =================================
    with PdfPages("UBL_KPI_Forecast_Report.pdf") as pdf:

        # ---------- Page 1: Narrative summary ----------
        fig, ax = plt.subplots(figsize=(11.7, 8.3))  # A4 landscape
        ax.axis("off")

        lines = []
        lines.append("UBL – ML-Based Financial KPI Forecast")
        lines.append("======================================")
        lines.append("")
        lines.append(
            f"History available from {kpi_hist.index.min().date()} "
            f"to {kpi_hist.index.max().date()} (quarterly)."
        )
        lines.append(
            f"Forecast horizon: {FUTURE_YEARS} years "
            f"({FUTURE_STEPS} quarters) beyond {kpi_hist.index.max().date()}."
        )
        lines.append("")
        lines.append("Models used per KPI:")
        lines.append("  • Naive random-walk benchmark")
        lines.append("  • SARIMAX (ARIMA with quarterly seasonality)")
        lines.append("  • Ridge regression on KPI lags")
        lines.append("Best model is selected by lowest backtest MAPE per KPI.")
        lines.append("")
        lines.append("Key messages (based on year-end levels):")

        hist_end_year = kpi_hist.index.year.max()
        fc_end_year = future_levels_df.index.year.max()

        for kpi in level_kpis:
            if kpi not in yearly_summary.columns:
                continue

            try:
                base_val = yearly_summary.loc[
                    yearly_summary.index.year == hist_end_year, kpi
                ].iloc[0]
                fut_val = yearly_summary.loc[
                    yearly_summary.index.year == fc_end_year, kpi
                ].iloc[0]
            except IndexError:
                continue

            if pd.isna(base_val) or pd.isna(fut_val) or base_val <= 0:
                continue

            years = fc_end_year - hist_end_year
            if years <= 0:
                continue

            cagr = (fut_val / base_val) ** (1 / years) - 1
            lines.append(
                f"  • {kpi}: from {base_val:,.0f} in {hist_end_year} "
                f"to {fut_val:,.0f} by {fc_end_year} "
                f"(~{cagr*100:,.1f}% CAGR)."
            )

        text = "\n".join(lines)
        ax.text(0.01, 0.99, text, va="top", fontsize=10)
        pdf.savefig()
        plt.close()

        # ---------- KPI charts (Actual vs Forecast) ----------
        for kpi in level_kpis:
            if kpi not in future_levels_df.columns:
                continue

            hist = kpi_hist[kpi].dropna()
            fut = future_levels_df[kpi]

            fig, ax = plt.subplots(figsize=(11.7, 4))
            ax.plot(hist.index, hist.values, label="Actual")
            ax.plot(fut.index, fut.values, "--", label="Forecast")

            # annotate last forecast value
            last_val = fut.iloc[-1]
            last_date = fut.index[-1]
            ax.annotate(
                f"{last_val:,.0f}",
                xy=(last_date, last_val),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

            ax.set_title(f"{kpi} – Actual vs Forecast")
            ax.set_xlabel("Date")
            ax.set_ylabel("Amount (Rupees in '000)")
            ax.legend()
            ax.grid(True)
            fig.tight_layout()
            pdf.savefig()
            plt.close()

        # ---------- Ratios charts ----------
        fig, axes = plt.subplots(3, 1, figsize=(11.7, 10), sharex=True)
        axes[0].plot(ratios_combined.index, ratios_combined["LDR"])
        axes[0].set_title("Loan-Deposit Ratio (Advances / Deposits)")
        axes[0].grid(True)

        axes[1].plot(ratios_combined.index, ratios_combined["Equity_to_Assets"])
        axes[1].set_title("Equity / Total Assets")
        axes[1].grid(True)

        axes[2].plot(ratios_combined.index, ratios_combined["ROE_Annualised_pct"])
        axes[2].set_title(
            "ROE – Annualised (based on change in Unappropriated Profit)"
        )
        axes[2].set_ylabel("%")
        axes[2].grid(True)

        fig.tight_layout()
        pdf.savefig()
        plt.close()

    print("📄 PDF saved as UBL_KPI_Forecast_Report.pdf")


if __name__ == "__main__":
    main()
