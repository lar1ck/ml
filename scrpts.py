
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# file_path = "loans.csv"
# df = pd.read_csv(file_path)

# df.columns = df.columns.str.strip()
# print("\n📊 Dataset Overview:")
# print("Cleaned column names:", df.columns.tolist())
# print("\nDataset shape:", df.shape)

# def clean_numeric_column(column):
#     if column.dtype == 'object':
#         return pd.to_numeric(column.str.replace(',', ''), errors='coerce')
#     return column

# features = [col for col in df.columns if col not in ['loan_id', 'loan_status']]
# print("\nSelected features:", features)

# categorical_features = ['education', 'self_employed']
# for col in categorical_features:
#     if col in df.columns:
#         df[col] = df[col].str.strip() 
#         if col == 'education':
#             df[col] = (df[col] == 'Graduate').astype(int)
#         elif col == 'self_employed':
#             df[col] = (df[col] == 'Yes').astype(int)

# numeric_features = [col for col in features if col not in categorical_features]
# for col in numeric_features:
#     df[col] = clean_numeric_column(df[col])

# target = 'loan_status'
# if target in df.columns:
#     print("\nTarget variable found:", target)
#     df[target] = df[target].str.strip()  
#     df[target] = (df[target] == 'Approved').astype(int)
#     print(f"Target variable distribution:\n{df[target].value_counts(normalize=True)}")
# else:
#     raise ValueError(f"Target column '{target}' not found in dataset")

# df = df.dropna(subset=features + [target])
# print("\nDataset shape after cleaning:", df.shape)


# X = df[features]
# y = df[target]

# print("\n📊 Dataset Splitting:")
# print(f"Total dataset size: {len(df)} samples")

# X_train, X_temp, y_train, y_temp = train_test_split(
#     X, y, 
#     test_size=0.2,
#     random_state=42,
#     stratify=y  
# )

# X_val, X_test, y_val, y_test = train_test_split(
#     X_temp, y_temp,
#     test_size=0.5,
#     random_state=42,
#     stratify=y_temp
# )

# print(f"\nTraining set: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
# print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(df)*100:.1f}%)")
# print(f"Test set: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")

# print("\n🤖 Training Model...")
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)

# y_train_pred = rf_model.predict(X_train)
# y_val_pred = rf_model.predict(X_val)

# def evaluate_model(y_true, y_pred, dataset_name):
#     print(f"\n📌 {dataset_name} Performance:")
#     print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_true, y_pred):.4f}")
#     print(f"Mean Squared Error (MSE): {mean_squared_error(y_true, y_pred):.4f}")
#     print(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
#     print(f"R² Score: {r2_score(y_true, y_pred):.4f}")

# evaluate_model(y_train, y_train_pred, "Training Set")
# evaluate_model(y_val, y_val_pred, "Validation Set")

# plt.figure(figsize=(10, 6))
# sns.scatterplot(x=y_val, y=y_val_pred, alpha=0.6)
# plt.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')
# plt.xlabel("Actual Loan Status (0=Rejected, 1=Approved)")
# plt.ylabel("Predicted Probability of Approval")
# plt.title("Actual vs. Predicted Loan Status")
# plt.legend()
# plt.tight_layout()
# plt.show()

# importances = pd.DataFrame({
#     'feature': features,
#     'importance': rf_model.feature_importances_
# }).sort_values('importance', ascending=True)

# plt.figure(figsize=(12, 6))
# sns.barplot(data=importances, x='importance', y='feature')
# plt.title("Feature Importance in Loan Prediction")
# plt.xlabel("Importance Score")
# plt.tight_layout()
# plt.show()

# print("\n📌 Dataset Statistics:")
# print(f"Total number of samples: {len(df)}")
# print("\nFeature Statistics:")
# print(df[features + [target]].describe())

# print("\n📌 Class Distribution:")
# print("Approval Rate:", (df[target] == 1).mean() * 100, "%")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# 1. Load dataset
file_path = "loans.csv"
df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.strip()

# Clean numeric columns
def clean_numeric_column(column):
    if column.dtype == 'object':
        return pd.to_numeric(column.str.replace(',', ''), errors='coerce')
    return column

features = [col for col in df.columns if col not in ['loan_id', 'loan_status']]

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
    df[target] = df[target].str.strip()
    df[target] = (df[target] == 'Approved').astype(int)
else:
    raise ValueError(f"Target column '{target}' not found in dataset")

df = df.dropna(subset=features + [target])

# 3. Split dataset
X = df[features]
y = df[target]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# 4. Initialize Models
rf_model = RandomForestRegressor(random_state=42)
gb_model = GradientBoostingRegressor(random_state=42)
logreg_model = LogisticRegression(random_state=42, solver='liblinear')

# 5. Fit the training data
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
logreg_model.fit(X_train, y_train)

# 6. Predict results
rf_train_pred = rf_model.predict(X_train)
rf_val_pred = rf_model.predict(X_val)
rf_test_pred = rf_model.predict(X_test)

gb_train_pred = gb_model.predict(X_train)
gb_val_pred = gb_model.predict(X_val)
gb_test_pred = gb_model.predict(X_test)

logreg_train_pred = logreg_model.predict(X_train)
logreg_val_pred = logreg_model.predict(X_val)
logreg_test_pred = logreg_model.predict(X_test)

# 7. Visualize Predictions (using Random Forest predictions for example)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_val, y=rf_val_pred, alpha=0.6)
plt.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')
plt.xlabel("Actual Loan Status")
plt.ylabel("Predicted Loan Status (RF)")
plt.title("Actual vs. Predicted Loan Status (RF)")
plt.legend()
plt.tight_layout()
plt.show()

# 8. Evaluation Metrics
def evaluate_classification(y_true, y_pred, dataset_name):
    y_pred_binary = (y_pred > 0.5).astype(int)
    print(f"\n📌 {dataset_name} Classification Metrics:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred_binary):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred_binary):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred_binary):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred_binary):.4f}")

def evaluate_regression(y_true, y_pred, dataset_name):
    print(f"\n📌 {dataset_name} Regression Metrics:")
    print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"R² Score: {r2_score(y_true, y_pred):.4f}")

# Evaluate Random Forest
print("\n--- Random Forest ---")
evaluate_regression(y_train, rf_train_pred, "RF Training Set")
evaluate_regression(y_val, rf_val_pred, "RF Validation Set")
evaluate_regression(y_test, rf_test_pred, "RF Test Set")
evaluate_classification(y_val, rf_val_pred, "RF Validation Set (Classification)")
evaluate_classification(y_test, rf_test_pred, "RF Test Set (Classification)")

# Evaluate Gradient Boosting
print("\n--- Gradient Boosting ---")
evaluate_regression(y_train, gb_train_pred, "GB Training Set")
evaluate_regression(y_val, gb_val_pred, "GB Validation Set")
evaluate_regression(y_test, gb_test_pred, "GB Test Set")
evaluate_classification(y_val, gb_val_pred, "GB Validation Set (Classification)")
evaluate_classification(y_test, gb_test_pred, "GB Test Set (Classification)")

# Evaluate Logistic Regression
print("\n--- Logistic Regression ---")
evaluate_classification(y_train, logreg_train_pred, "LR Training Set")
evaluate_classification(y_val, logreg_val_pred, "LR Validation Set")
evaluate_classification(y_test, logreg_test_pred, "LR Test Set")

# 10. Tuning Hyperparameters (GridSearchCV)
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=3, scoring='r2', verbose=1, n_jobs=-1)
grid_search_rf.fit(X_train, y_train)
best_rf_model = grid_search_rf.best_estimator_

rf_val_pred_tuned = best_rf_model.predict(X_val)
rf_test_pred_tuned = best_rf_model.predict(X_test)

# 11, 12. Apply Evaluation Metrics on Validation and Test Data (Tuned Model)
print("\n--- Tuned Random Forest ---")
evaluate_regression(y_val, rf_val_pred_tuned, "Tuned RF Validation Set")
evaluate_regression(y_test, rf_test_pred_tuned, "Tuned RF Test Set")
evaluate_classification(y_val, rf_val_pred_tuned, "Tuned RF Validation Set (Classification)")
evaluate_classification(y_test, rf_test_pred_tuned, "Tuned RF Test Set (Classification)")

# 13. Overfitting/Underfitting Detection (using R2 score for example)
print("\n--- Overfitting/Underfitting Detection ---")
print(f"RF Training R2: {r2_score(y_train, rf_train_pred):.4f}")
print(f"RF Validation R2: {r2_score(y_val, rf_val_pred):.4f}")
print(f"RF Test R2: {r2_score(y_test, rf_test_pred):.4f}")

print(f"GB Training R2: {r2_score(y_train, gb_train_pred):.4f}")
print(f"GB Validation R2: {r2_score(y_val, gb_val_pred):.4f}")
print(f"GB Test R2: {r2_score(y_test, gb_test_pred):.4f}")

# 14. Re-evaluate on Validation and Test Data (After Overfitting/Underfitting Analysis)
# (If necessary, you would adjust the model or hyperparameters based on the analysis from step 13)

# Interpretation of the model:
# Feature Importance for Random Forest
importances = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importances (Random Forest):")
print(importances)

# Interpretation of Logistic Regression coefficients
print("\nLogistic Regression Coefficients:")
print(pd.DataFrame({'Feature': X.columns, 'Coefficient': logreg_model.coef_[0]}))

# New data(unseen data) prediction example. Assuming new data is in a dataframe called new_data
# create a sample new_data dataframe
new_data = pd.DataFrame({
    'gender': ['Male', 'Female'],
    'married': ['Yes', 'No'],
    'dependents': [0, 1],
    'education': ['Graduate', 'Not Graduate'],
    'self_employed': ['No', 'Yes'],
    'applicantincome': [5000, 6000],
    'coapplicantincome': [1000, 0],
    'loanamount': [120, 150],
    'loan_amount_term': [360, 180],
    'credit_history': [1, 0],
    'property_area': ['Urban', 'Rural']
})

# Preprocessing new_data to be consistent with training data
new_data['education'] = (new_data['education'] == 'Graduate').astype(int)
new_data['self_employed'] = (new_data['self_employed'] == 'Yes').astype(int)

# One hot encode categorical variables.
new_data = pd.get_dummies(new_data, columns=['gender','married','property_area'], drop_first=True)

# Align columns to match training data
new_data = new_data.reindex(columns=X_train.columns, fill_value=0)

# Predict using the trained model
rf_new_predictions = rf_model.predict(new_data)
logreg_new_predictions = logreg_model.predict(new_data)

print("\nRandom Forest Predictions for New Data:")
print(rf_new_predictions)

print("\nLogistic Regression Predictions for New Data:")
print(logreg_new_predictions)