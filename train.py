import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("C:/Users/shyak/OneDrive/Documents/L5NOTES/Machine Learning/ww2.csv")

print(df.head())

# Function to handle numeric ranges (e.g., "4,440,000 to 5,318,000")
def parse_numeric_range(value):
    if isinstance(value, str):
        parts = value.replace(',', '').split(' to ')
        if len(parts) == 2:
            return (float(parts[0]) + float(parts[1])) / 2  # Take the average
        elif len(parts) == 1:
            return float(parts[0])  # Single value
    return np.nan  # If not convertible, return NaN

# Apply function to relevant columns
numeric_columns = [
    "Total population as of 1/1/1939", "Militarydeaths from all causes",
    "Civilian deaths due to military activity and crimes against humanity",
    "Totaldeaths", "Militarywounded"
]

for col in numeric_columns:
    df[col] = df[col].astype(str).apply(parse_numeric_range)

# Fill missing values with 0 (or other strategies)
df.fillna(0, inplace=True)

X = df[["Total population as of 1/1/1939", "Civilian deaths due to military activity and crimes against humanity", "Totaldeaths"]]
y = df["Militarydeaths from all causes"]

# Split dataset: 60% train, 20% validation, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Print sizes
print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")

model = LinearRegression()
model.fit(X_train, y_train)

# Validate model
y_val_pred = model.predict(X_val)
val_mae = mean_absolute_error(y_val, y_val_pred)
print(f"Validation MAE: {val_mae:.2f}")

# Test model
y_test_pred = model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_test_pred)
print(f"Test MAE: {test_mae:.2f}")