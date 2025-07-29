import pandas as pd

# Load CSVs
nba_df = pd.read_csv("data/nba_workload.csv")
nfl_df = pd.read_csv("data/nfl_workload.csv")
mlb_df = pd.read_csv("data/mlb_workload.csv")
injury_df = pd.read_csv("data/injury_labels.csv")

# Print previews of each dataset
print("NBA Data:\n", nba_df.head(), "\n")
print("NFL Data:\n", nfl_df.head(), "\n")
print("MLB Data:\n", mlb_df.head(), "\n")
print("Injury Labels:\n", injury_df.head())

# Add sport_type to each dataframe
nba_df["sport_type"] = "NBA"
nfl_df["sport_type"] = "NFL"
mlb_df["sport_type"] = "MLB"

# Rename workload columns
nba_df = nba_df.rename(columns={"minutes_played": "workload"})
nfl_df = nfl_df.rename(columns={"snaps": "workload"})
mlb_df = mlb_df.rename(columns={"innings_pitched": "workload"})

# Combine datasets
combined_df = pd.concat([nba_df, nfl_df, mlb_df], ignore_index=True)

# Merge with injury labels
merged_df = pd.merge(combined_df, injury_df, on=["player_id", "season"], how="left")

# Preview final merged data
print("\nCombined Data with Injury Labels:\n")
print(merged_df.head())

# Drop 'injured' missing rows
clean_df = merged_df.dropna(subset=["injured"])

# Convert 'injured' float to int
clean_df["injured"] = clean_df["injured"].astype(int)

# Preview cleaned data
print("\nCleaned Data (no missing labels):\n")
print(clean_df.head())

# One-hot encode 'position' and 'sport_type'
encoded_df = pd.get_dummies(clean_df, columns=["position", "sport_type"])

# Preview final encoded dataframe
print("\nEncoded Data:\n")
print(encoded_df.head())

from sklearn.model_selection import train_test_split

# Drop non-numeric and ID columns features
X = encoded_df.drop(columns=["player_id", "season", "injured"])
y = encoded_df["injured"]

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining set size:", X_train.shape[0])
print("Testing set size:", X_test.shape[0])

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # For ROC-AUC

# Evaluate the model
print("\n--- Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]

print("\n--- Random Forest ---")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall:", recall_score(y_test, rf_pred))
print("F1 Score:", f1_score(y_test, rf_pred))
print("ROC-AUC:", roc_auc_score(y_test, rf_proba))

# Train XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

print("\n--- XGBoost ---")
print("Accuracy:", accuracy_score(y_test, xgb_pred))
print("Precision:", precision_score(y_test, xgb_pred))
print("Recall:", recall_score(y_test, xgb_pred))
print("F1 Score:", f1_score(y_test, xgb_pred))
print("ROC-AUC:", roc_auc_score(y_test, xgb_proba))
