# Injury_Risk_Prediction

This project predicts athlete injury risk using supervised machine learning models applied to player workload data across the NBA, NFL, and MLB.

## Project Overview

Using real player workload metrics (e.g., minutes played, snaps, innings pitched, rest days, and age), this model attempts to predict injury occurrence.  
The goal is to help sports organizations and trainers proactively identify at-risk athletes and make informed decisions.

##  Technologies Used

- **Python** (3.13)
- **pandas**, **NumPy** — data cleaning and manipulation
- **scikit-learn**, **XGBoost** — building classification models
- **matplotlib**, **seaborn** — performance visualization
- **Tableau** — interactive charts and dashboard visualizations
- **VS Code** — development environment

## Machine Learning Models

The model compares three supervised classification methods:
- **Logistic Regression**
- **Random Forest**
- **XGBoost**

Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

## Data

- Synthetic but realistically structured data was used to simulate workload patterns in each league.
- Dataset includes:
  - **NBA**: Minutes played, age, rest days
  - **NFL**: Snaps, age, rest days
  - **MLB**: Innings pitched, age, rest days
  - **Injury Labels** (binary classification: injured = 1, not injured = 0)

> Note: Real datasets will be integrated in the next iteration.

## How it Works

1. Loads and merges sport-specific workload data
2. Cleans and encodes features
3. Trains models to classify injury risk
4. Evaluates performance using ROC-AUC and other metrics

## Results (So Far)

- Best performance from **Random Forest** model
- ROC-AUC Score: `0.6667`
- F1-Score: `0.80`
- Precision: `0.80`
- Accuracy: `75%`

> More tuning and real-world data integration is planned.

## Future Work

- Use real sports datasets (public or scraped)
- Find model with best outcome to use
- Add visualizations with finalized data
- Improve model generalizability with more seasons/teams

## Author

**Lindsey George**  
Senior CS Student @ Bowie State University  
[LinkedIn](https://www.linkedin.com/in/lindsey-george-13a32a252)

---

