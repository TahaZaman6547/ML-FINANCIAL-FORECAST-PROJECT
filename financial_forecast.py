import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from matplotlib.backends.backend_pdf import PdfPages
import shap

np.random.seed(123)

# =========================
# 1. CREATE SYNTHETIC DATA
# =========================

dates = pd.date_range(start="2015-01-01", periods=40, freq="Q")
t = np.arange(len(dates))

GDP = 1000 + 5*t + np.random.normal(0, 10, len(t))
CPI = 100 + 0.6*t + np.random.normal(0, 2, len(t))
RATE = 2 + 0.02*t + np.random.normal(0, 0.2, len(t))

GDP_growth = pd.Series(GDP).pct_change().fillna(0)
CPI_growth = pd.Series(CPI).pct_change().fillna(0)

revenue = 500 + 4*t + 60*GDP_growth - 40*RATE + np.random.normal(0, 25, len(t))

data = pd.DataFrame({
    "date": dates,
    "revenue": revenue,
    "GDP_growth": GDP_growth,
    "CPI_growth": CPI_growth,
    "RATE": RATE
}).set_index("date")


# =========================
# 2. FEATURE ENGINEERING
# =========================

for lag in [1,2,3,4]:
    data[f"rev_lag{lag}"] = data["revenue"].shift(lag)

data["rev_roll4"] = data["revenue"].rolling(4).mean()
data = data.dropna()

feature_cols = [
    "rev_lag1","rev_lag2","rev_lag3","rev_roll4",
    "GDP_growth","CPI_growth","RATE"
]

X = data[feature_cols]
y = data["revenue"]

test_horizon = 12   # 3 years ahead

X_train = X.iloc[:-test_horizon]
X_test  = X.iloc[-test_horizon:]
y_train = y.iloc[:-test_horizon]
y_test  = y.iloc[-test_horizon:]


# =========================
# 3. MODELS
# =========================

sarimax = SARIMAX(
    y_train,
    exog=X_train,
    order=(1,1,1),
    seasonal_order=(1,1,1,4)
).fit(disp=False)

sarimax_fc = sarimax.get_forecast(steps=test_horizon, exog=X_test)
sarimax_pred = sarimax_fc.predicted_mean

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)

rf = RandomForestRegressor(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)


# =========================
# 4. EVALUATION
# =========================

def mape(a,p):
    return np.mean(np.abs((a-p)/a)) * 100

models_perf = pd.DataFrame({
    "Model":["SARIMAX","Ridge","RandomForest"],
    "MAE":[
        mean_absolute_error(y_test, sarimax_pred),
        mean_absolute_error(y_test, ridge_pred),
        mean_absolute_error(y_test, rf_pred)
    ],
    "RMSE":[
        np.sqrt(mean_squared_error(y_test, sarimax_pred)),
        np.sqrt(mean_squared_error(y_test, ridge_pred)),
        np.sqrt(mean_squared_error(y_test, rf_pred))
    ],
    "MAPE":[
        mape(y_test, sarimax_pred),
        mape(y_test, ridge_pred),
        mape(y_test, rf_pred)
    ]
})

print("\nModel Performance:")
print(models_perf)


# =========================
# 5. FORECAST TABLE
# =========================

forecast_df = pd.DataFrame({
    "date": y_test.index,
    "Actual": y_test.values,
    "SARIMAX": sarimax_pred.values,
    "Ridge": ridge_pred,
    "RF": rf_pred
}).set_index("date")


# =========================
# 6. ANNUAL FORECAST SUMMARY
# =========================

annual = forecast_df[["SARIMAX"]].copy()
annual["Year"] = annual.index.year
annual_totals = annual.groupby("Year").sum()

print("\nAnnual Forecast Summary:")
print(annual_totals)


# =========================
# 7. SAVE EXCEL
# =========================

with pd.ExcelWriter("Financial_Forecast_Output.xlsx") as writer:
    data.to_excel(writer, sheet_name="Historical_Data")
    forecast_df.to_excel(writer, sheet_name="Quarterly_Forecasts")
    annual_totals.to_excel(writer, sheet_name="Annual_Forecast")
    models_perf.to_excel(writer, sheet_name="Model_Performance")

print("\nExcel saved as: Financial_Forecast_Output.xlsx")


# =========================
# 8. CREATE NARRATIVE
# =========================

text_lines = [
    "Financial Forecast Narrative (Synthetic Data)",
    "----------------------------------------------",
]

for year, val in annual_totals["SARIMAX"].items():
    text_lines.append(f"In {year}, expected forecasted revenue ≈ {val:,.0f} units.")

narrative_text = "\n".join(text_lines)

print("\n" + narrative_text)


# =========================
# 9. SAVE PDF REPORT
# =========================

with PdfPages("Financial_Forecast_Report.pdf") as pdf:

    plt.figure(figsize=(12,5))
    plt.plot(forecast_df.index, forecast_df["Actual"], label="Actual", marker="o")
    plt.plot(forecast_df.index, forecast_df["SARIMAX"], label="Forecast", marker="x")
    plt.title("Revenue Forecast")
    plt.legend()
    pdf.savefig()
    plt.close()

    fig, ax = plt.subplots(figsize=(8,4))
    ax.axis('off')
    table = ax.table(cellText=models_perf.values, colLabels=models_perf.columns, loc='center')
    ax.set_title("Model Performance", pad=20)
    pdf.savefig()
    plt.close()

    fig, ax = plt.subplots(figsize=(8,4))
    ax.axis('off')
    ax.text(0, 1, narrative_text, fontsize=12, va="top")
    pdf.savefig()
    plt.close()

print("PDF saved as: Financial_Forecast_Report.pdf")
