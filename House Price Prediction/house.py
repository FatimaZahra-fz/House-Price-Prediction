# ============================================
# California Housing Price Prediction Project
# ============================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================
# 1. LOAD DATASET
# ============================================

# Use your correct CSV file path
df = pd.read_csv(r"C:\Users\HP\Desktop\housing.csv")

# Show first 5 rows
print("===== FIRST 5 ROWS =====")
print(df.head())

# ============================================
# 2. DATASET INFORMATION
# ============================================

print("\n===== DATASET INFO =====")
print(df.info())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())

# ============================================
# 3. HANDLE MISSING VALUES
# ============================================

# Fill missing values in total_bedrooms
df['total_bedrooms'].fillna(
    df['total_bedrooms'].median(),
    inplace=True
)

# ============================================
# 4. FEATURE SELECTION
# ============================================

# Independent Variables (Features)
X = df[[
    'longitude',
    'latitude',
    'housing_median_age',
    'total_rooms',
    'total_bedrooms',
    'population',
    'households',
    'median_income'
]]

# Dependent Variable (Target)
y = df['median_house_value']

# ============================================
# 5. SPLIT DATASET
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape :", X_test.shape)

# ============================================
# 6. TRAIN MODEL
# ============================================

model = LinearRegression()

print("\nTraining Model...")
model.fit(X_train, y_train)

print("Model Training Completed!")

# ============================================
# 7. MODEL COEFFICIENTS
# ============================================

print("\n===== MODEL COEFFICIENTS =====")

for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")

print("\nIntercept:")
print(model.intercept_)

# ============================================
# 8. MAKE PREDICTIONS
# ============================================

y_pred = model.predict(X_test)

# ============================================
# 9. MODEL EVALUATION
# ============================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\n===== MODEL EVALUATION =====")

print(f"Mean Absolute Error (MAE): {mae:.2f}")

print(f"Mean Squared Error (MSE): {mse:.2f}")

print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

print(f"R2 Score: {r2:.4f}")

# ============================================
# 10. ACTUAL VS PREDICTED VALUES
# ============================================

comparison = pd.DataFrame({
    'Actual Price': y_test.values,
    'Predicted Price': y_pred
})

print("\n===== ACTUAL VS PREDICTED =====")
print(comparison.head(10))

# ============================================
# 11. VISUALIZATION
# ============================================

# Correlation Heatmap
plt.figure(figsize=(12, 8))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title("Correlation Heatmap")

plt.show()

# Actual vs Predicted Scatter Plot
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Prices")

plt.ylabel("Predicted Prices")

plt.title("Actual vs Predicted House Prices")

plt.show()

# ============================================
# 12. PREDICT NEW HOUSE PRICE
# ============================================

new_house = pd.DataFrame({
    'longitude': [-122.23],
    'latitude': [37.88],
    'housing_median_age': [20],
    'total_rooms': [2500],
    'total_bedrooms': [500],
    'population': [1000],
    'households': [400],
    'median_income': [5.0]
})

predicted_price = model.predict(new_house)

print("\n===== PREDICTED HOUSE PRICE =====")

print(f"Predicted Price: ${predicted_price[0]:,.2f}")

# ============================================
# END OF PROJECT
# ============================================