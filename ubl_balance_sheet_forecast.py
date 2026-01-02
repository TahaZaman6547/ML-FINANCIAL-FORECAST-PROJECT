import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from matplotlib.backends.backend_pdf import PdfPages


# ==============================
# CONFIG
# ==============================

DATA_PATH = "Data/UBL_Financials_.xlsx"
DATE_COL = "Date"

TEST_STEPS = 4          # last 4 quarters = backtest
FUTURE_STEPS = 12       # 12 quarters = 3 years


TARGET_VARS = [
    "Cash and balances with treasury banks",
    "Balances with other banks",
    "Investments",
    "Advances",
    "Property and equipment",
    "Other assets",
    "Bills payable",
    "Borrowings",
    "Deposits and other accounts",
    "Lease liabilities",
    "Subordinated debt",
    "Deferred tax liabilities",
    "Other liabilities",
    "Share capital",
    "Reserves",
    "Surplus on revaluation of assets",
    "Unappropriated profit"
]


# ==============================
# LOAD & CLEAN DATA
# ==============================

def load_data():

    df = pd.read_excel(DATA_PATH)

    df.columns = [str(c).strip() for c in df.columns]

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    df = df.sort_values(DATE_COL)

    # remove duplicates safely
    if df[DATE_COL].duplicated().any():
        print("⚠ Duplicate dates detected → keeping last")
        df = df.drop_duplicates(subset=[DATE_COL], keep="last")

    df = df.set_index(DATE_COL)

    # force quarterly frequency
    df = df.asfreq("Q-DEC")

    # numeric only
    df = df.select_dtypes(include=["number"])

    df = df.ffill()

    return df



# ==============================
# FEATURE ENGINEERING
# ==============================

def make_xy(df, target):

    d = df.copy()
    d["target"] = d[target]

    d["lag1"] = d[target].shift(1)
    d["lag2"] = d[target].shift(2)
    d["lag3"] = d[target].shift(3)

    d = d.dropna()

    X = d[["lag1","lag2","lag3"]]
    y = d["target"]

    return X,y



# ==============================
# MODEL A SINGLE VARIABLE
# ==============================

def model_target(df, target):

    X,y = make_xy(df,target)

    X_train = X.iloc[:-TEST_STEPS]
    y_train = y.iloc[:-TEST_STEPS]

    X_test = X.iloc[-TEST_STEPS:]
    y_test = y.iloc[-TEST_STEPS:]

    # ---- SARIMAX ----
    sar = SARIMAX(y_train, order=(1,1,1)).fit(disp=False)
    sar_pred = sar.get_forecast(TEST_STEPS).predicted_mean

    # ---- Ridge ----
    rg = Ridge(alpha=1.0)
    rg.fit(X_train,y_train)
    rg_pred = rg.predict(X_test)

    # ---- RF ----
    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    rf.fit(X_train,y_train)
    rf_pred = rf.predict(X_test)

    def mape(a,p): 
        return np.mean(np.abs((a-p)/a))*100

    metrics = pd.DataFrame({
        "Model":["SARIMAX","Ridge","RandomForest"],
        "MAE":[
            mean_absolute_error(y_test,sar_pred),
            mean_absolute_error(y_test,rg_pred),
            mean_absolute_error(y_test,rf_pred)
        ],
        "RMSE":[
            np.sqrt(mean_squared_error(y_test,sar_pred)),
            np.sqrt(mean_squared_error(y_test,rg_pred)),
            np.sqrt(mean_squared_error(y_test,rf_pred))
        ],
        "MAPE":[
            mape(y_test,sar_pred),
            mape(y_test,rg_pred),
            mape(y_test,rf_pred)
        ]
    })

    backtest = pd.DataFrame({
        "Actual":y_test.values,
        "SARIMAX":sar_pred.values,
        "Ridge":rg_pred,
        "RF":rf_pred
    }, index=y_test.index)

    # ---- Future Forecast ----
    last_vals = y.values[-3:].tolist()
    future_dates = pd.date_range(y.index[-1]+pd.offsets.QuarterEnd(),
                                 periods=FUTURE_STEPS,
                                 freq="Q")

    preds = []
    for _ in range(FUTURE_STEPS):
        x = np.array(last_vals[-3:]).reshape(1,-1)
        val = rf.predict(x)[0]
        preds.append(val)
        last_vals.append(val)

    future = pd.DataFrame({"Forecast_RF":preds}, index=future_dates)

    return metrics, backtest, future



# ==============================
# MAIN
# ==============================

def main():

    df = load_data()
    print("Loaded rows:", len(df))

    all_metrics = []
    all_future = {}

    for t in TARGET_VARS:
        if t not in df.columns:
            print(f"❌ Missing column: {t}")
            continue
        
        print(f"→ Modelling: {t}")

        m, bt, fut = model_target(df,t)
        m["Target"] = t
        all_metrics.append(m)
        all_future[t] = fut

    metrics_all = pd.concat(all_metrics)

    # =======================
    # SAVE EXCEL
    # =======================

    with pd.ExcelWriter("UBL_Forecast_Output.xlsx") as w:

        df.to_excel(w, sheet_name="Input_Data")
        metrics_all.to_excel(w, sheet_name="Model_Performance")

        for t,f in all_future.items():
            f.to_excel(w, sheet_name=f"Future_{t[:25]}")

    print("📊 Excel Saved")

    # =======================
    # SAVE PDF
    # =======================

    with PdfPages("UBL_Forecast_Report.pdf") as pdf:

        for t,f in all_future.items():
            plt.figure(figsize=(10,4))
            plt.plot(f.index,f["Forecast_RF"],marker="o")
            plt.title(f"{t} — Future Forecast")
            plt.grid()
            pdf.savefig()
            plt.close()

    print("📄 PDF Saved")



if __name__ == "__main__":
    main()
