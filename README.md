# House Price Prediction: California Housing (CodeNova AI/ML Batch 02, Task 3)

End-to-end regression pipeline predicting median house value for California census block groups.

## Tech stack
Python, pandas, NumPy, scikit-learn, Matplotlib, joblib. Free and 100% local.

## Setup
```bash
pip install -r requirements.txt
python train.py
```
Outputs: `results.csv`, `model.joblib`, and charts in `charts/`.

## Methodology
- **Data:** 20,640 rows, 9 numeric features plus `ocean_proximity` (categorical). `total_bedrooms` has missing values.
- **Feature engineering:** `rooms_per_household`, `bedrooms_per_room`, `population_per_household`. These use only input columns, never the target.
- **No leakage:** the 80/20 train-test split happens *before* anything is fit. Median imputation, scaling and one-hot encoding live inside a scikit-learn `Pipeline`, so they are fit on training folds only.
- **Models:** mean-predicting baseline, Ridge, Random Forest (200 trees). 5-fold CV on the training set, final scores on the held-out test set.

## Results (test set)
| Model | CV RMSE | Test RMSE | Test MAE | Test R² |
|---|---|---|---|---|
| Baseline (mean) | 115,619 | 114,486 | 90,607 | 0.000 |
| Ridge | 67,848 | 69,135 | 49,652 | 0.635 |
| **Random Forest** | 50,126 | 49,454 | 31,746 | **0.813** |

Random Forest cuts RMSE by about 57% versus the baseline. CV and test scores agree closely, so the model is not overfitting the test set.

**Top features:** median income (0.49), inland location (0.14), population per household (0.12), longitude and latitude (~0.06 each).

Charts: `charts/actual_vs_predicted.png`, `charts/feature_importance.png`, `charts/residuals.png`.

## Limitations and failure cases
- **Target capping:** 965 rows have values capped at $500,001, so the model can't learn true prices of expensive homes and under-predicts them (see residual chart). MAE on those capped rows is about $75.9k versus $31.7k overall.
- **Old data:** this is 1990 census data. Prices and relationships have shifted, so it should not be used for real valuation.
- **Block-group level:** features are neighborhood averages, not individual house attributes (size, condition).
- **Geography:** the model uses raw latitude/longitude with no spatial validation, so errors may be correlated between nearby areas.
- **Untuned:** Random Forest uses reasonable defaults. Hyperparameter search (e.g. gradient boosting) could improve it further.

## Dataset
California Housing (Pace and Barry, 1997), as mirrored in the *Hands-On Machine Learning* repo.
