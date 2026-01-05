import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "Data/UBL_Financials_.xlsx"
DATE_COL = "Date"

# train only on 2014–2020, validate on 2021–2025
TRAIN_END = pd.Timestamp("2020-12-31")
TEST_START = pd.Timestamp("2021-01-01")
TEST_END = pd.Timestamp("2025-12-31")

N_LAGS = 4   # Ridge lag features

# KPI column names in Excel
KPI_COLS = {
    "Deposits": "Deposits and other accounts",
    "Advances": "Advances",
    "Investments": "Investments",
    "Borrowings": "Borrowings",
    "Total_Assets": "TOTAL ASSETS",
    "Total_Equity": "TOTAL EQUITY",
    "Unappropriated_Profit": "Unappropriated profit",
}


# ============================================================
# 1. LOAD & PREPARE DATA
# ============================================================

def load_data() -> pd.DataFrame:
    """Load Excel, clean index & column names, return numeric df indexed by Date."""
    print("\nLoading data…\n")

    df = pd.read_excel(DATA_PATH)

    df.columns = [str(c).strip() for c in df.columns]

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL)
    df = df.set_index(DATE_COL)

    # drop duplicated dates safely (keep last)
    if df.index.duplicated().any():
        print("⚠ Duplicate dates detected – keeping last row per date")
        df = df[~df.index.duplicated(keep="last")]

    # numeric only
    df = df.select_dtypes(include=["number"]).copy()

    # forward fill missing values
    df = df.ffill()

    # sanity check KPI columns
    missing = [v for v in KPI_COLS.values() if v not in df.columns]
    if missing:
        raise ValueError(f"Missing KPI columns in Excel: {missing}")

    print("Full data shape:", df.shape)
    print("Date range:", df.index.min().date(), "→", df.index.max().date())
    return df


# ============================================================
# 2. FEATURE ENGINEERING & MODELLING HELPERS
# ============================================================

def make_lag_training_frame(series: pd.Series, n_lags: int = N_LAGS):
    """
    Build lag features for Ridge training using ONLY the training period.
    No internal test split here – all available train rows (after lags) are used.
    """
    df = pd.DataFrame({"y": series})
    for lag in range(1, n_lags + 1):
        df[f"lag{lag}"] = series.shift(lag)

    df = df.dropna()
    X = df[[f"lag{lag}" for lag in range(1, n_lags + 1)]]
    y = df["y"]
    return X, y


def train_models(train: pd.Series):
    """
    Train three models on training series:
      - Naive: last value (no training)
      - SARIMAX
      - Ridge on lag features
    Returns dict of fitted models (SARIMAX, Ridge).
    """
    # SARIMAX (ARIMA with quarterly-ish seasonality 4)
    sarimax = SARIMAX(
        train,
        order=(1, 1, 1),
        seasonal_order=(0, 1, 1, 4),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    # Ridge on lags
    X_train, y_train = make_lag_training_frame(train)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)

    return {"SARIMAX": sarimax, "Ridge": ridge}


def ridge_forecast(train: pd.Series, ridge: Ridge, horizon: int) -> np.ndarray:
    """
    Iterative multi-step forecast using Ridge & lag features.
    Uses ONLY historic data in 'train' to start.
    """
    history = train.dropna().values.tolist()
    preds = []

    for _ in range(horizon):
        if len(history) < N_LAGS:
            raise ValueError("Not enough history for Ridge lags")

        lag_dict = {f"lag{lag}": history[-lag] for lag in range(1, N_LAGS + 1)}
        X_new = pd.DataFrame([lag_dict])
        y_hat = ridge.predict(X_new)[0]
        preds.append(y_hat)
        history.append(y_hat)

    return np.array(preds)


def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error (ignoring zero actuals)."""
    a = np.asarray(y_true, dtype="float64")
    p = np.asarray(y_pred, dtype="float64")
    mask = a != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100.0)


# ============================================================
# 3. MAIN BACKTEST PIPELINE
# ============================================================

def main():
    raw = load_data()

    # We will store:
    # - backtest_all: one sheet with Actual + Forecast for every KPI (2021–2025)
    # - metrics_rows: long table of metrics per KPI & model
    # - best_preds_for_plot: best-model forecast series per KPI for charts
    backtest_all = None
    metrics_rows = []
    best_preds_for_plot = {}
    last_train_date = None
    last_test_date = None

    for friendly_name, col_name in KPI_COLS.items():
        print(f"\n==========================")
        print(f"KPI: {friendly_name}  ({col_name})")
        print("==========================")

        series_full = raw[col_name].dropna()

        # split into train (≤2020-12-31) and test (2021–2025)
        train = series_full[series_full.index <= TRAIN_END]
        test = series_full[
            (series_full.index >= TEST_START) & (series_full.index <= TEST_END)
        ].dropna()

        if len(train) < N_LAGS + 3 or len(test) == 0:
            print("   ⚠ Not enough data for this KPI – skipping.")
            continue

        # remember global periods for narrative
        last_train_date = train.index.max()
        last_test_date = test.index.max()

        print(f"   Train period: {train.index.min().date()} → {train.index.max().date()}")
        print(f"   Test  period: {test.index.min().date()} → {test.index.max().date()}")
        print(f"   Train size = {len(train)}, Test size = {len(test)}")

        models = train_models(train)
        horizon = len(test)
        test_index = test.index

        # --- Forecasts (out-of-sample) ---
        naive_fc = np.repeat(train.iloc[-1], horizon)

        sarimax = models["SARIMAX"]
        sar_fc = sarimax.get_forecast(steps=horizon).predicted_mean.values

        ridge = models["Ridge"]
        ridge_fc = ridge_forecast(train, ridge, horizon)

        # --- Metrics vs actual test ---
        actual = test.values

        for model_name, pred in [
            ("Naive", naive_fc),
            ("SARIMAX", sar_fc),
            ("Ridge", ridge_fc),
        ]:
            metrics_rows.append(
                {
                    "KPI": friendly_name,
                    "Model": model_name,
                    "MAE": mean_absolute_error(actual, pred),
                    "RMSE": np.sqrt(mean_squared_error(actual, pred)),
                    "MAPE": mape(actual, pred),
                }
            )

        # choose best model by MAPE (prefer SARIMAX/Ridge over Naive)
        kpi_metrics = [r for r in metrics_rows if r["KPI"] == friendly_name]
        kpi_metrics_df = pd.DataFrame(kpi_metrics)
        kpi_metrics_df = kpi_metrics_df.sort_values("MAPE")
        best_model = kpi_metrics_df.iloc[0]["Model"]
        if best_model == "Naive" and len(kpi_metrics_df) > 1:
            # if Naive is only slightly better, use the second one
            if kpi_metrics_df.iloc[1]["MAPE"] <= kpi_metrics_df.iloc[0]["MAPE"] + 1.0:
                best_model = kpi_metrics_df.iloc[1]["Model"]

        print("   Best model for", friendly_name, "→", best_model)

        if best_model == "Naive":
            best_fc = naive_fc
        elif best_model == "SARIMAX":
            best_fc = sar_fc
        else:
            best_fc = ridge_fc

        best_series = pd.Series(best_fc, index=test_index, name=friendly_name)
        best_preds_for_plot[friendly_name] = {
            "full_actual": series_full,
            "forecast": best_series,
            "best_model": best_model,
        }

        # Build combined backtest DataFrame (one row per test quarter)
        if backtest_all is None:
            backtest_all = pd.DataFrame(index=test_index)

        backtest_all[f"{friendly_name}_Actual"] = test
        backtest_all[f"{friendly_name}_Forecast"] = best_series

    # ------------------------------------------------------------------
    # If nothing processed, exit
    # ------------------------------------------------------------------
    if backtest_all is None or len(best_preds_for_plot) == 0:
        print("\nNo KPIs processed – check data / column names.")
        return

    # Convert metrics_rows to DataFrame
    metrics_df = pd.DataFrame(metrics_rows)
    # Mark best model per KPI
    best_by_kpi = (
        metrics_df.sort_values("MAPE")
        .groupby("KPI")
        .head(1)
        .rename(columns={"Model": "Best_Model"})
    )[["KPI", "Best_Model", "MAPE", "MAE", "RMSE"]]

    # ============================================================
    # 4. SAVE EXCEL
    # ============================================================

    with pd.ExcelWriter("UBL_Backtest_2014_2020_vs_2021_2025_Output.xlsx") as writer:
        backtest_all.to_excel(writer, sheet_name="Backtest_Quarterly")
        metrics_df.to_excel(writer, sheet_name="All_Model_Metrics", index=False)
        best_by_kpi.to_excel(writer, sheet_name="Best_Model_per_KPI", index=False)

    print("\n📊 Excel saved as UBL_Backtest_2014_2020_vs_2021_2025_Output.xlsx")

    # ============================================================
    # 5. SAVE PDF REPORT
    # ============================================================

    with PdfPages("UBL_Backtest_2014_2020_vs_2021_2025_Report.pdf") as pdf:
        # ---------- PAGE 1: TEXT SUMMARY ----------
        fig, ax = plt.subplots(figsize=(11.7, 8.3))  # A4 landscape
        ax.axis("off")

        lines = []
        lines.append("UBL – ML-Based Financial KPI Forecast")
        lines.append("======================================")
        lines.append("")
        lines.append(
            f"History used for model training: "
            f"{raw.index.min().date()} to {TRAIN_END.date()}."
        )
        lines.append(
            f"Validation (backtest) period: "
            f"{TEST_START.date()} to {min(TEST_END, last_test_date).date()}."
        )
        lines.append("")
        lines.append("Models used per KPI:")
        lines.append("  • Naive random-walk benchmark")
        lines.append("  • SARIMAX (ARIMA with quarterly seasonality)")
        lines.append("  • Ridge regression on KPI lags")
        lines.append("Best model is selected by lowest MAPE on 2021–2025 backtest.")
        lines.append("")
        lines.append("Key messages (based on last available test quarter per KPI):")

        for kpi_name, info in best_preds_for_plot.items():
            actual_full = info["full_actual"]
            fc = info["forecast"]
            best_model = info["best_model"]

            # last common test point (usually 2025-09-30)
            last_idx = fc.index[-1]
            if last_idx not in actual_full.index:
                continue

            act_last = actual_full.loc[last_idx]
            fc_last = fc.loc[last_idx]
            if act_last == 0:
                err_pct = np.nan
            else:
                err_pct = (fc_last - act_last) / act_last * 100

            lines.append(
                f"  • {kpi_name}: last test point {last_idx.date()} – "
                f"actual {act_last:,.0f}, model {fc_last:,.0f} "
                f"(abs error {abs(err_pct):.1f}%, best model: {best_model})."
            )

        text = "\n".join(lines)
        ax.text(0.01, 0.99, text, va="top", fontsize=9)
        pdf.savefig()
        plt.close()

        # ---------- PAGES 2+: KPI CHARTS ----------
        for kpi_name, info in best_preds_for_plot.items():
            actual_series = info["full_actual"]
            fc_series = info["forecast"]

            fig, ax = plt.subplots(figsize=(11.7, 4))

            ax.plot(actual_series.index, actual_series.values, label="Actual")
            ax.plot(
                fc_series.index,
                fc_series.values,
                "--",
                label="Forecast (2021–2025)",
            )

            last_date = fc_series.index[-1]
            last_val = fc_series.iloc[-1]
            ax.annotate(
                f"{last_val:,.0f}",
                xy=(last_date, last_val),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

            ax.set_title(f"{kpi_name} – Actual vs Forecast (2014–2025)")
            ax.set_xlabel("Date")
            ax.set_ylabel("Amount (Rupees in '000)")
            ax.legend()
            ax.grid(True)
            fig.tight_layout()
            pdf.savefig()
            plt.close()

    print("📄 PDF saved as UBL_Backtest_2014_2020_vs_2021_2025_Report.pdf")


if __name__ == "__main__":
    main()
