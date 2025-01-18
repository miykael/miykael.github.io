import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    PolynomialFeatures,
    StandardScaler,
    RobustScaler,
    PowerTransformer,
)
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance
from sklearn.model_selection import ParameterGrid, GridSearchCV, KFold, cross_validate
from pathlib import Path


# Set random seed for reproducibility
np.random.seed(42)

# Create output directory for figures if it doesn't exist
plot_dir = Path("../assets/ex_plots")
plot_dir.mkdir(parents=True, exist_ok=True)

# 1. Load Dataset
housing = datasets.fetch_openml(name="house_prices")
X = housing["data"].drop(columns="Id")
y = housing["target"]

print(f"Dimension of X: {X.shape}\nDimension of y: {y.shape}")

# Show first few entries
print("\nFirst few entries:")
print(X.iloc[:5, :5])

# Feature analysis
print("\nFeature types:")
print(X.dtypes.value_counts())

missing_values = X.isnull().sum()
print("\nFeatures with missing values:")
print(missing_values[missing_values > 0].sort_values(ascending=False))

# Visualize target distribution
plt.figure(figsize=(8, 4))
sns.histplot(y, bins=50)
plt.title("Distribution of House Prices")
plt.xlabel("Price")
plt.savefig(
    plot_dir / "03_scikit_price_distribution.png",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
plt.close()

# 3. Split data
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)

# 4. Build Pipeline
# Categorical preprocessing
categorical_preprocessor = Pipeline(
    [
        ("imputer_cat", SimpleImputer(fill_value="missing", strategy="constant")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Feature reduction pipeline
dim_reduction = FeatureUnion(
    [
        ("pca", PCA()),
        ("feat_selecter", SelectKBest()),
    ]
)

# Numerical preprocessing
numeric_preprocessor = Pipeline(
    [
        ("imputer_numeric", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ("scaler", StandardScaler()),
        ("polytrans", PolynomialFeatures(degree=2, include_bias=False)),
        ("dim_reduction", dim_reduction),
    ]
)

# Combine preprocessors
preprocessor = ColumnTransformer(
    [
        ("numerical", numeric_preprocessor, X.select_dtypes("number").columns),
        (
            "categorical",
            categorical_preprocessor,
            X.select_dtypes(exclude="number").columns,
        ),
    ],
    remainder="passthrough",
)

# Create full pipeline with Ridge regression
pipe = Pipeline(steps=[("preprocessor", preprocessor), ("ridge", Ridge())])

# Add target transformation
regressor = TransformedTargetRegressor(
    regressor=pipe, func=np.log1p, inverse_func=np.expm1
)

# 5. Define parameter grid
prefix = "regressor__preprocessor__"
param_grid = {
    f"{prefix}numerical__imputer_numeric__add_indicator": [True, False],
    f"{prefix}numerical__imputer_numeric__strategy": [
        "mean",
        "median",
        "most_frequent",
        "constant",
    ],
    f"{prefix}categorical__imputer_cat__add_indicator": [True, False],
    f"{prefix}categorical__imputer_cat__strategy": ["most_frequent", "constant"],
    f"{prefix}numerical__polytrans__degree": [1, 2],
    f"{prefix}numerical__polytrans__interaction_only": [False, True],
    f"{prefix}numerical__dim_reduction__pca": ["drop", PCA(0.9), PCA(0.99)],
    f"{prefix}numerical__dim_reduction__feat_selecter__k": [5, 25, 100, "all"],
    f"{prefix}numerical__dim_reduction__feat_selecter__score_func": [
        f_regression,
        mutual_info_regression,
    ],
    f"{prefix}numerical__scaler": [
        StandardScaler(),
        RobustScaler(),
        PowerTransformer(),
    ],
    "regressor__ridge__alpha": np.logspace(-5, 5, 11),
}

print("\nTotal parameter combinations:", len(ParameterGrid(param_grid)))

# 6. Train model using RandomizedSearchCV
random_search = RandomizedSearchCV(
    regressor,
    param_grid,
    n_iter=250,
    refit=True,
    cv=2,
    return_train_score=True,
    n_jobs=-1,
    verbose=1,
    scoring="neg_mean_absolute_percentage_error",
)

res = random_search.fit(X_tr, y_tr)

# 7. Performance investigation
df_res = pd.DataFrame(res.cv_results_)
df_res = df_res.iloc[:, ~df_res.columns.str.contains("time|split[0-9]*|rank|params")]
new_columns = [
    c.split("param_regressor__")[1] if "param_regressor" in c else c
    for c in df_res.columns
]
new_columns = [
    c.split("preprocessor__")[1] if "preprocessor__" in c else c for c in new_columns
]
df_res.columns = new_columns
df_res = df_res.sort_values("mean_test_score", ascending=False)

print("\nTop 10 parameter combinations:")
print(df_res.head(10))

# Evaluate model performance
score_tr = -random_search.score(X_tr, y_tr)
score_te = -random_search.score(X_te, y_te)

print(f"\nPrediction accuracy on train data: {score_tr * 100:.2f}%")
print(f"Prediction accuracy on test data:  {score_te * 100:.2f}%")

# 8. Fine tune best preprocessing pipeline
best_estimator = random_search.best_estimator_
param_grid = {"regressor__ridge__alpha": np.logspace(-5, 5, 51)}

# Nested cross-validation

inner_cv = KFold(n_splits=3, shuffle=True)
outer_cv = KFold(n_splits=3, shuffle=True)

grid_search = GridSearchCV(
    best_estimator,
    param_grid,
    refit=True,
    cv=inner_cv,
    n_jobs=-1,
    scoring="neg_mean_absolute_percentage_error",
)

cv_results = cross_validate(
    grid_search,
    X=X,
    y=y,
    cv=outer_cv,
    n_jobs=1,
    return_estimator=True,
    return_train_score=True,
)

# Print cross-validation results
df_nested = pd.DataFrame(cv_results)
cv_train_scores = -df_nested["train_score"]
cv_test_scores = -df_nested["test_score"]
cv_alphas = [c.best_params_["regressor__ridge__alpha"] for c in df_nested["estimator"]]

print("\nGeneralization score with hyperparameters tuning:")
print(
    f"  Train Score:    {cv_train_scores.mean() * 100:.1f}% +/- {cv_train_scores.std() * 100:.1f}%"
)
print(
    f"  Test Score:     {cv_test_scores.mean() * 100:.1f}% +/- {cv_test_scores.std() * 100:.1f}%"
)
print(f"  Optimal Alpha: {np.mean(cv_alphas):.1f} +/- {np.std(cv_alphas):.1f}")

# 9. Feature importance investigation
final_estimator = grid_search.set_params(
    estimator__regressor__ridge__alpha=np.mean(cv_alphas)
)
_ = final_estimator.fit(X_tr, y_tr)

scoring = ["r2", "neg_mean_absolute_percentage_error"]
result = permutation_importance(
    final_estimator,
    X_te,
    y_te,
    n_repeats=50,
    random_state=0,
    n_jobs=-1,
    scoring=scoring,
)

# Plot feature importance
fig, axs = plt.subplots(1, 2, figsize=(16, 16))
for i, s in enumerate(scoring):
    sorted_idx = result[s].importances_mean.argsort()
    axs[i].boxplot(
        result[s].importances[sorted_idx].T, vert=False, labels=X_te.columns[sorted_idx]
    )
    axs[i].set_title(f"Permutation Importances (test set) | {s}")
plt.tight_layout()
plt.savefig(
    plot_dir / "03_scikit_feature_importance.png",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
plt.close()
