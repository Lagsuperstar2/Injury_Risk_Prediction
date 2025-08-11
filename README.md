# Injury_Risk_Prediction

This project predicts athlete injury risk using supervised machine learning applied to player workload data across the NBA, NFL, and MLB.

## Project Overview

Using sport-specific workload metrics (e.g., minutes played, snaps, pitch-based workload, rest days, and age), this model predicts whether an athlete is at risk of injury.  
The goal is to help sports organizations and trainers make data-driven decisions to reduce injury likelihood.

## Technologies Used

- **Python** (3.13)
- **pandas**, **NumPy** — data cleaning and manipulation  
- **scikit-learn** — building and evaluating the classification model  
- **Tableau** — creating visualizations and dashboards  
- **VS Code** — development environment

## Machine Learning Model

- **Logistic Regression** (with class balancing and feature scaling)

**Evaluation Metrics**:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

## Data

- **Synthetic but realistic datasets** representing workload and injury patterns across three sports.
- Dataset includes:
  - **NBA**: Minutes played, age, rest days
  - **NFL**: Snaps (converted to workload), age, rest days
  - **MLB**: Pitch-based workload (estimated from innings pitched), age, rest days
  - **Injury Labels** (binary classification: injured = 1, not injured = 0)

> Note: Real-world datasets will be integrated in future iterations.

## How It Works

1. Loads and merges league-specific workload data
2. Standardizes column names and cleans missing values
3. Adds new features:
   - **workload_per_age** = workload ÷ age
   - **rest_per_workload** = rest days ÷ (workload + 1)
4. Encodes categorical variables (`sport_type`, `position`)
5. Splits into training/testing sets with stratification
6. Scales numeric features
7. Trains a Logistic Regression model
8. Evaluates performance using multiple metrics

## Results (Final Submission)

- Accuracy: **0.75**  
- Precision: **0.80**  
- Recall: **0.80**  
- F1-Score: **0.80**  
- ROC-AUC: **0.78**

## Future Work

- Integrate real historical sports datasets
- Explore ensemble methods (Random Forest, XGBoost) for comparison
- Add injury severity prediction instead of binary classification
- Improve model generalizability with additional seasons and player stats

## Author

**Lindsey George**  
Senior Computer Science Student @ Bowie State University  
[LinkedIn](https://www.linkedin.com/in/lindsey-george-13a32a252)
