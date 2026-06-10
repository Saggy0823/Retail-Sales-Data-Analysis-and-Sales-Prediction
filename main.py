import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ====================================
# CREATE RETAIL DATASET
# ====================================

np.random.seed(42)

months = np.arange(1, 25)

sales = [
    1200,1300,1280,1400,1450,1500,
    1600,1700,1680,1750,1800,1900,
    2000,2100,2200,2300,2400,2500,
    2600,2700,2800,2900,3000,3200
]

df = pd.DataFrame({
    "Month": months,
    "Sales": sales
})

print("\n===== DATASET =====")
print(df.head())

# ====================================
# STATISTICAL SUMMARY
# ====================================

print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())

# ====================================
# SALES TREND
# ====================================

plt.figure(figsize=(10,5))
plt.plot(df["Month"], df["Sales"], marker='o')
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.grid(True)
plt.show()

# ====================================
# CORRELATION
# ====================================

plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# ====================================
# MACHINE LEARNING MODEL
# ====================================

X = df[["Month"]]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

# ====================================
# EVALUATION
# ====================================

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n===== MODEL PERFORMANCE =====")
print("Mean Absolute Error:", round(mae,2))
print("R2 Score:", round(r2,2))

# ====================================
# FUTURE SALES PREDICTION
# ====================================

future_months = pd.DataFrame({
    "Month":[25,26,27,28,29,30]
})

future_sales = model.predict(future_months)

future_df = pd.DataFrame({
    "Month": future_months["Month"],
    "Predicted Sales": future_sales
})

print("\n===== FUTURE SALES PREDICTION =====")
print(future_df)

# ====================================
# VISUALIZATION
# ====================================

plt.figure(figsize=(10,5))

plt.scatter(
    X,
    y,
    label="Actual Sales"
)

plt.plot(
    X,
    model.predict(X),
    linewidth=2,
    label="Regression Line"
)

plt.title("Sales Prediction Model")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()

plt.show()

# ====================================
# BUSINESS INSIGHTS
# ====================================

print("\n===== BUSINESS INSIGHTS =====")

print("1. Sales show a strong upward trend.")
print("2. Revenue increases steadily every month.")
print("3. High positive correlation exists between time and sales.")
print("4. Future sales are expected to continue growing.")
print("5. Inventory planning should account for increasing demand.")

print("\nProject Completed Successfully!")
