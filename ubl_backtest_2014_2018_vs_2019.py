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

KPI_FILE = "Data/UBL_Financials_.xlsx"
MACRO_FILE = "Data/macro_quarterly.xlsx"
DATE_COL = "Date"

# We will forecast these years one-by-one
FORECAST_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

N_LAGS = 4

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
# HELPERS
# ============================================================

def load_kpi_data():
    df = pd.read_excel(KPI_FILE)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.set_index(DATE_COL).sort_index()

    if df.index.duplicated().any():
        print("⚠ KPI data has duplicate dates – keeping last row per date")
        df = df[~df.index.duplicated(keep="last")]

    df = df.select_dtypes(include=["number"]).ffill()

    print(f"KPI data shape: {df.shape}")
    print(f"KPI date range: {df.index.min().date()} → {df.index.max().date()}")
    return df


def load_macro_data():
    df = pd.read_excel(MACRO_FILE)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.set_index(DATE_COL).sort_index()

    if df.index.duplicated().any():
        print("⚠ Macro data has duplicate dates – keeping last row per date")
        df = df[~df.index.duplicated(keep="last")]

    df = df.select_dtypes(include=["number"]).ffill()

    print(f"Macro data shape: {df.shape}")
    print(f"Macro date range: {df.index.min().date()} → {df.index.max().date()}")
    return df


def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def make_lag_features(series):
    df = pd.DataFrame({"y": series})
    for lag in range(1, N_LAGS + 1):
        df[f"lag{lag}"] = series.shift(lag)
    return df.dropna()


def ridge_forecast(series, horizon):
    df = make_lag_features(series)
    if len(df) == 0:
        return np.array([np.nan] * horizon)

    X = df[[f"lag{i}" for i in range(1, N_LAGS + 1)]]
    y = df["y"]

    model = Ridge(alpha=1.0)
    model.fit(X, y)

    history = list(series.dropna())
    preds = []

    for _ in range(horizon):
        row = [history[-i] for i in range(1, N_LAGS + 1)]
        X_new = pd.DataFrame([row], columns=X.columns)
        y_hat = model.predict(X_new)[0]
        preds.append(y_hat)
        history.append(y_hat)

    return np.array(preds)


# ============================================================
# MAIN
# ============================================================

def main():
    kpi = load_kpi_data()
    macro = load_macro_data()

    all_backtests = {yr: {} for yr in FORECAST_YEARS}
    metrics_rows = []

    for friendly_name, col_name in KPI_COLS.items():
        print("\n==============================")
        print(f"KPI: {friendly_name}  ({col_name})")
        print("==============================")

        if col_name not in kpi.columns:
            continue

        series = kpi[col_name].dropna()

        # 🔥 enforce uniqueness always before merge
        series = series[~series.index.duplicated(keep="last")]
        macro_clean = macro[~macro.index.duplicated(keep="last")]

        df = pd.concat([series, macro_clean], axis=1, join="inner").dropna()
        df.columns = ["KPI", "CPI", "GDP", "PolicyRate", "USD_PKR"]

        if df.empty:
            continue

        for year in FORECAST_YEARS:
            train = df[df.index.year <= (year - 1)]
            test = df[df.index.year == year]

            if len(train) == 0 or len(test) == 0:
                continue

            horizon = len(test)

            # models
            naive_fc = np.repeat(train["KPI"].iloc[-1], horizon)

            try:
                sar = SARIMAX(
                    train["KPI"],
                    exog=train[["CPI", "GDP", "PolicyRate", "USD_PKR"]],
                    order=(1, 1, 1),
                    seasonal_order=(0, 1, 1, 4),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)

                sar_fc = sar.get_forecast(
                    steps=horizon,
                    exog=test[["CPI", "GDP", "PolicyRate", "USD_PKR"]],
                ).predicted_mean.values

            except Exception:
                sar_fc = np.full(horizon, np.nan)

            ridge_fc = ridge_forecast(train["KPI"], horizon)

            actual = test["KPI"].values

            models = {
                "Naive": naive_fc,
                "SARIMAX_X": sar_fc,
                "Ridge_Lag": ridge_fc,
            }

            metric_list = []
            for m, pred in models.items():
                metric_list.append({
                    "Year": year,
                    "KPI": friendly_name,
                    "Model": m,
                    "MAE": mean_absolute_error(actual, pred),
                    "RMSE": np.sqrt(mean_squared_error(actual, pred)),
                    "MAPE": mape(actual, pred),
                })

            metrics_rows.extend(metric_list)

            best_model = sorted(metric_list, key=lambda d: d["MAPE"])[0]["Model"]
            best_pred = models[best_model]

            bt = pd.DataFrame({
                "Actual": actual,
                "Naive": naive_fc,
                "SARIMAX_X": sar_fc,
                "Ridge_Lag": ridge_fc,
                "Best": best_pred,
            }, index=test.index)

            all_backtests[year][friendly_name] = bt

    # ================= SAVE EXCEL =================

    metrics_df = pd.DataFrame(metrics_rows)

    with pd.ExcelWriter("UBL_Rolling_Backtests_2014_2025_Output.xlsx") as writer:
        for year in FORECAST_YEARS:
            if not all_backtests[year]:
                continue

            combined = pd.DataFrame(
                index=sorted({i for bt in all_backtests[year].values() for i in bt.index})
            )

            for kpi_name, bt in all_backtests[year].items():
                for col in bt.columns:
                    combined[f"{kpi_name}_{col}"] = bt[col].reindex(combined.index)

            combined.to_excel(writer, sheet_name=f"Backtest_{year}")

        metrics_df.to_excel(writer, sheet_name="All_Model_Metrics", index=False)

    print("\n📊 Excel saved as UBL_Rolling_Backtests_2014_2025_Output.xlsx")

    # ================= SAVE PDF =================

    with PdfPages("UBL_Rolling_Backtests_2014_2025_Report.pdf") as pdf:
        for year in FORECAST_YEARS:
            for kpi_name, bt in all_backtests[year].items():
                fig, ax = plt.subplots(figsize=(11.7, 4))
                ax.plot(bt.index, bt["Actual"], label="Actual")
                ax.plot(bt.index, bt["Best"], "--", label="Forecast")
                ax.set_title(f"{kpi_name} – {year}")
                ax.legend()
                ax.grid(True)
                pdf.savefig()
                plt.close()

    print("📄 PDF saved as UBL_Rolling_Backtests_2014_2025_Report.pdf")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
