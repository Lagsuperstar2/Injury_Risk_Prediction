import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

print("✅ Script started")

# Load datasets
print("📂 Loading NBA CSV...")
nba_df = pd.read_csv("data/nba_workload.csv", encoding='latin1')
print("✅ NBA CSV loaded")

print("📂 Loading NFL CSV...")
nfl_df = pd.read_csv("data/nfl_workload.csv", encoding='latin1')
print("✅ NFL CSV loaded")

print("📂 Loading MLB CSV...")
mlb_df = pd.read_csv("data/mlb_workload.csv", encoding='latin1')
print("✅ MLB CSV loaded")

# Standardize and clean
nba_df["rest_days"] = np.random.choice([1, 2, 3, 4], size=len(nba_df))
nba_df["sport_type"] = "NBA"
nba_df = nba_df[["player", "player_id", "season", "workload", "rest_days", "age", "position", "injured", "sport_type"]]

nfl_df = nfl_df.rename(columns={"Player": "player", "snaps": "workload", "Injured": "injured"})
nfl_df["sport_type"] = "NFL"
nfl_df = nfl_df[["player", "player_id", "season", "workload", "rest_days", "age", "position", "injured", "sport_type"]]

mlb_df = mlb_df.rename(columns={"Player": "player", "Injured": "injured"})
mlb_df["sport_type"] = "MLB"
mlb_df = mlb_df[["player", "player_id", "season", "workload", "rest_days", "age", "position", "injured", "sport_type"]]

# Combine
combined_df = pd.concat([nba_df, nfl_df, mlb_df], ignore_index=True)
combined_df = combined_df.dropna(subset=["injured"])
combined_df["injured"] = combined_df["injured"].astype(int)

# Feature engineering
combined_df["workload_per_age"] = combined_df["workload"] / combined_df["age"]
combined_df["rest_per_workload"] = combined_df["rest_days"] / (combined_df["workload"] + 1)

# One-hot encoding
categorical_cols = ["sport_type", "position"]
encoded_df = pd.get_dummies(combined_df[categorical_cols], drop_first=True)

# Prepare final dataset
features_df = pd.concat([
    combined_df.drop(columns=["player_id", "player", "injured"] + categorical_cols),
    encoded_df
], axis=1)

# Drop remaining NaNs
features_df = features_df.dropna()
combined_df = combined_df.loc[features_df.index].reset_index(drop=True)

# Save player info
player_ids = combined_df["player_id"].reset_index(drop=True)
player_names = combined_df["player"].reset_index(drop=True)

# Define X and y
X = features_df
y = combined_df["injured"]

# Print class distribution
print("\nInjury Class Distribution:")
print(y.value_counts())
print("\n% Distribution:")
print(y.value_counts(normalize=True) * 100)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
lr = LogisticRegression(max_iter=1000, class_weight="balanced")
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)

print("\n--- Logistic Regression Only ---")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print("Precision:", precision_score(y_test, lr_pred))
print("Recall:", recall_score(y_test, lr_pred))
print("F1 Score:", f1_score(y_test, lr_pred))
print("ROC-AUC:", roc_auc_score(y_test, lr_pred))

# Show sample predictions
rf_results = pd.DataFrame({
    "player_id": player_ids.iloc[y_test.index].values,
    "player_name": player_names.iloc[y_test.index].values,
    "actual_injured": y_test.values,
    "predicted_lr": lr_pred
})

print("\n--- Sample Predictions ---")
print(rf_results.head())
