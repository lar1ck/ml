
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

file_path = "loans.csv"
df = pd.read_csv(file_path)

df.columns = df.columns.str.strip()
print("\n📊 Dataset Overview:")
print("Cleaned column names:", df.columns.tolist())
print("\nDataset shape:", df.shape)

def clean_numeric_column(column):
    if column.dtype == 'object':
        return pd.to_numeric(column.str.replace(',', ''), errors='coerce')
    return column

features = [col for col in df.columns if col not in ['loan_id', 'loan_status']]
print("\nSelected features:", features)

categorical_features = ['education', 'self_employed']
for col in categorical_features:
    if col in df.columns:
        df[col] = df[col].str.strip() 
        if col == 'education':
            df[col] = (df[col] == 'Graduate').astype(int)
        elif col == 'self_employed':
            df[col] = (df[col] == 'Yes').astype(int)

numeric_features = [col for col in features if col not in categorical_features]
for col in numeric_features:
    df[col] = clean_numeric_column(df[col])

target = 'loan_status'
if target in df.columns:
    print("\nTarget variable found:", target)
    df[target] = df[target].str.strip()  
    df[target] = (df[target] == 'Approved').astype(int)
    print(f"Target variable distribution:\n{df[target].value_counts(normalize=True)}")
else:
    raise ValueError(f"Target column '{target}' not found in dataset")

df = df.dropna(subset=features + [target])
print("\nDataset shape after cleaning:", df.shape)


X = df[features]
y = df[target]

print("\n📊 Dataset Splitting:")
print(f"Total dataset size: {len(df)} samples")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42,
    stratify=y  
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

print(f"\nTraining set: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(df)*100:.1f}%)")
print(f"Test set: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")

print("\n🤖 Training Model...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_train_pred = rf_model.predict(X_train)
y_val_pred = rf_model.predict(X_val)

def evaluate_model(y_true, y_pred, dataset_name):
    print(f"\n📌 {dataset_name} Performance:")
    print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"Mean Squared Error (MSE): {mean_squared_error(y_true, y_pred):.4f}")
    print(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"R² Score: {r2_score(y_true, y_pred):.4f}")

evaluate_model(y_train, y_train_pred, "Training Set")
evaluate_model(y_val, y_val_pred, "Validation Set")

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_val, y=y_val_pred, alpha=0.6)
plt.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')
plt.xlabel("Actual Loan Status (0=Rejected, 1=Approved)")
plt.ylabel("Predicted Probability of Approval")
plt.title("Actual vs. Predicted Loan Status")
plt.legend()
plt.tight_layout()
plt.show()

importances = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=True)

plt.figure(figsize=(12, 6))
sns.barplot(data=importances, x='importance', y='feature')
plt.title("Feature Importance in Loan Prediction")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

print("\n📌 Dataset Statistics:")
print(f"Total number of samples: {len(df)}")
print("\nFeature Statistics:")
print(df[features + [target]].describe())

print("\n📌 Class Distribution:")
print("Approval Rate:", (df[target] == 1).mean() * 100, "%")