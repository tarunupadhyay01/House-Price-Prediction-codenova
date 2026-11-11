"""Task 3: House Price Prediction (California Housing). Run: python train.py"""
import numpy as np, pandas as pd, joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

SEED = 42
df = pd.read_csv("housing.csv")
# Feature engineering uses only row-level features (no target) -> no target leakage
df["rooms_per_household"] = df["total_rooms"] / df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
df["population_per_household"] = df["population"] / df["households"]

target = "median_house_value"
X, y = df.drop(columns=target), df[target]
cat = ["ocean_proximity"]
num = [c for c in X.columns if c not in cat]

# Split FIRST; imputer/scaler/encoder are fit inside pipeline on training folds only
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)

def prep(scale=True):
    steps = [("imp", SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if scale else [])
    return ColumnTransformer([("num", Pipeline(steps), num),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])

models = {
    "Baseline (mean)": Pipeline([("p", prep()), ("m", DummyRegressor(strategy="mean"))]),
    "Ridge": Pipeline([("p", prep()), ("m", Ridge(alpha=1.0))]),
    "Random Forest": Pipeline([("p", prep(False)), ("m", RandomForestRegressor(
        n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=SEED))]),
}

rows, preds = [], {}
for name, pipe in models.items():
    cv = -cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    pipe.fit(X_tr, y_tr); p = pipe.predict(X_te); preds[name] = p
    rows.append(dict(Model=name, CV_RMSE=cv.mean(), CV_std=cv.std(),
        Test_RMSE=np.sqrt(mean_squared_error(y_te, p)),
        Test_MAE=mean_absolute_error(y_te, p), Test_R2=r2_score(y_te, p)))
res = pd.DataFrame(rows).round(3); print(res.to_markdown(index=False))
res.to_csv("results.csv", index=False)

best = models["Random Forest"]; joblib.dump(best, "model.joblib")

# Chart 1: actual vs predicted
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_te, preds["Random Forest"], s=6, alpha=.35)
lim = [0, y.max()]; ax.plot(lim, lim, "r--", label="Perfect prediction")
ax.set(title="Random Forest: Actual vs Predicted House Value", xlabel="Actual ($)", ylabel="Predicted ($)")
ax.legend(); fig.tight_layout(); fig.savefig("charts/actual_vs_predicted.png", dpi=200)

# Chart 2: feature importance
names = best["p"].get_feature_names_out()
imp = pd.Series(best["m"].feature_importances_, index=[n.split("__")[1] for n in names]).sort_values().tail(12)
fig, ax = plt.subplots(figsize=(7, 5)); imp.plot.barh(ax=ax)
ax.set(title="Random Forest Feature Importance (top 12)", xlabel="Importance"); fig.tight_layout()
fig.savefig("charts/feature_importance.png", dpi=200)
print(imp.sort_values(ascending=False).head(5).round(3))

# Chart 3: residuals by value (failure analysis)
res_ = y_te - preds["Random Forest"]
fig, ax = plt.subplots(figsize=(7, 4)); ax.scatter(y_te, res_, s=6, alpha=.35); ax.axhline(0, color="r")
ax.set(title="Residuals vs Actual Value", xlabel="Actual ($)", ylabel="Residual ($)"); fig.tight_layout()
fig.savefig("charts/residuals.png", dpi=200)
print("capped at 500001:", (y == 500001).sum(), "of", len(y))
print("MAE on capped rows:", mean_absolute_error(y_te[y_te>=500001], preds["Random Forest"][y_te>=500001]))
