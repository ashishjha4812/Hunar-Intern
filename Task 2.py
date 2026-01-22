# 1. Import libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 2. Load dataset

df = pd.read_csv(r"E:\Users\oma\Downloads\house price data.csv")

print("First 5 rows:")
print(df.head())
print("\nInfo:")
print(df.info())
print("\nMissing values:")
print(df.isna().sum())
print("\nDuplicates:", df.duplicated().sum())

# 3. Data preprocessing

# 3.1 Remove duplicate rows
df = df.drop_duplicates()

# 3.2 Handle missing values
# Separate numeric and categorical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(exclude=['int64', 'float64']).columns

# Fill numeric NaNs with median (robust to outliers)
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Fill categorical NaNs with mode
for col in cat_cols:
    mode_val = df[col].mode()
    if not mode_val.empty:
        df[col] = df[col].fillna(mode_val[0])

print("\nMissing values after filling:")
print(df.isna().sum())

# 4. Define features (X) and target (y)



target_col = "price"

if target_col not in df.columns:
    raise ValueError(f"Change target_col variable – '{target_col}' not found in columns: {df.columns}")

y = df[target_col]

# Drop target from features
X = df.drop(columns=[target_col])

# Convert categorical columns to numeric (one-hot encoding)
X = pd.get_dummies(X, drop_first=True)

print("\nShape of X (features):", X.shape)
print("Shape of y (target):", y.shape)

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# 6. Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Evaluation on test set
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print("\nModel Performance on Test Set:")
print("R² Score       :", r2)
print("MSE            :", mse)
print("RMSE           :", rmse)
print("MAE            :", mae)

# 8. Test with a single example from test set
sample = X_test.iloc[[0]]
true_price = y_test.iloc[0]
pred_price = model.predict(sample)[0]

print("\nExample prediction:")
print("Actual price   :", true_price)
print("Predicted price:", pred_price)
