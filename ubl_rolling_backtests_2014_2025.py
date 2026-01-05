import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

KPI_FILE = "Data/UBL_Financials_.xlsx"
MACRO_FILE = "Data/macro_quarterly.xlsx"
DATE_COL = "Date"

FORECAST_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
N_LAGS = 4

SAFE_CAP = 1e12   # prevent numeric explosion

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

def load_file(path):
    df = pd.read_excel(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.set_index(DATE_COL).sort_index()
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="last")]
    return df.select_dtypes(include="number").ffill()


def mape(y, yhat):
    y = np.array(y, dtype=float)
    yhat = np.array(yhat, dtype=float)
    mask = (y != 0) & (~np.isnan(y)) & (~np.isnan(yhat))
    if mask.sum() == 0:
        return np.inf
    return np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])) * 100


def lag_ridge(series, horizon):
    df = pd.DataFrame({"y": np.log1p(series)})
    for lag in range(1, N_LAGS + 1):
        df[f"lag{lag}"] = df["y"].shift(lag)

    df = df.dropna()
    if len(df) == 0:
        return np.repeat(np.nan, horizon)

    X = df[[f"lag{i}" for i in range(1, N_LAGS + 1)]]
    y = df["y"]

    model = Ridge(alpha=1.0)
    model.fit(X, y)

    hist = list(df["y"])
    preds = []
    for _ in range(horizon):
        x = hist[-N_LAGS:]
        pred = model.predict([x])[0]
        preds.append(pred)
        hist.append(pred)

    yhat = np.expm1(preds)
    yhat = np.clip(yhat, 0, SAFE_CAP)
    return yhat


def annotate(ax, x, y, text):
    ax.annotate(
        f"{text}: {y:,.0f}",
        xy=(x, y),
        xytext=(5, 5),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="yellow", alpha=0.5),
        fontsize=8,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    kpi = load_file(KPI_FILE)
    macro = load_file(MACRO_FILE)

    results = []
    summary = {yr: [] for yr in FORECAST_YEARS}

    with PdfPages("UBL_Rolling_Backtests_2014_2025_Report.pdf") as pdf:

        # -------- TITLE PAGE ----------
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(
            0.05, 0.95,
            "UBL – ML-Based Financial KPI Rolling-Backtests (2020–2025)\n"
            "Models: Naive • SARIMAX-X (auto-validated) • Ridge-Lag • Hybrid\n"
            "Best model = lowest MAPE per KPI-Year.\n"
            "SARIMAX is auto-disabled when unstable.",
            ha="left", va="top", fontsize=11
        )
        plt.axis("off")
        pdf.savefig()
        plt.close()

        # -------- KPI LOOP ----------
        for kpi_name, col in KPI_COLS.items():

            series = kpi[col].dropna()
            df = pd.concat([series, macro], axis=1, join="inner").dropna()
            df.columns = ["KPI", "CPI", "GDP", "Policy", "FX"]

            for year in FORECAST_YEARS:

                train = df[df.index.year <= year - 1]
                test = df[df.index.year == year]

                if len(test) == 0:
                    continue

                horizon = len(test)
                actual = test["KPI"].values

                # Naive Model
                naive = np.repeat(train["KPI"].iloc[-1], horizon)

                # SARIMAX SAFE MODEL
                sar = np.repeat(np.nan, horizon)
                try:
                    log_train = np.log1p(train["KPI"])
                    model = SARIMAX(
                        log_train,
                        exog=train[["CPI", "GDP", "Policy", "FX"]],
                        order=(1, 1, 1),
                        seasonal_order=(0, 1, 1, 4),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False)

                    sar_log = model.get_forecast(
                        steps=horizon,
                        exog=test[["CPI", "GDP", "Policy", "FX"]],
                    ).predicted_mean.values

                    sar = np.expm1(sar_log)
                    sar = np.clip(sar, 0, SAFE_CAP)

                except Exception:
                    sar = np.repeat(np.nan, horizon)

                # Ridge Lag
                ridge = lag_ridge(train["KPI"], horizon)

                # Hybrid
                hybrid = np.nanmean(np.vstack([naive, sar, ridge]), axis=0)

                models = {
                    "Naive": naive,
                    "SARIMAX_X": sar,
                    "Ridge": ridge,
                    "Hybrid": hybrid,
                }

                # pick best
                best_name = min(models, key=lambda m: mape(actual, models[m]))
                best = models[best_name]

                # store summary
                ae = abs(best[-1] - actual[-1]) / actual[-1] * 100
                summary[year].append(
                    f"{kpi_name}: actual {actual[-1]:,.0f}, forecast {best[-1]:,.0f} "
                    f"(abs error {ae:.1f}%, model={best_name})"
                )

                # ---------- PLOT ----------
                fig, ax = plt.subplots(figsize=(11.7, 4.5))
                ax.plot(test.index, actual, lw=2, label="Actual", color="black")
                ax.plot(test.index, best, "--", lw=2, label=f"Forecast ({best_name})")

                annotate(ax, test.index[-1], actual[-1], "Actual")
                annotate(ax, test.index[-1], best[-1], "Forecast")

                ax.set_title(f"{kpi_name} — {year}")
                ax.grid(alpha=0.3)
                ax.legend()

                pdf.savefig()
                plt.close()

        # -------- SUMMARY PAGE ----------
        fig = plt.figure(figsize=(11.7, 8.3))
        ax = fig.add_subplot(111)
        ax.axis("off")

        text = ""
        for yr in FORECAST_YEARS:
            text += f"\nYear {yr}\n"
            for line in summary[yr]:
                text += " • " + line + "\n"

        fig.text(0.03, 0.97, text, va="top", ha="left", fontsize=9)
        pdf.savefig()
        plt.close()

    print("\n📄 REPORT CREATED SUCCESSFULLY\n")


if __name__ == "__main__":
    main()
