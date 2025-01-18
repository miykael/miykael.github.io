import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Create output directory for figures if it doesn't exist
plot_dir = Path("../assets/ex_plots")
plot_dir.mkdir(parents=True, exist_ok=True)

# 1. Load dataset
digits = datasets.load_digits()
X = digits["data"]
y = digits["target"]

print(f"Dimension of X: {X.shape}\nDimension of y: {y.shape}")

# Visualize sample digits
_, axes = plt.subplots(nrows=3, ncols=8, figsize=(8, 4))
for ax, image, label in zip(axes.ravel(), digits.images, digits.target):
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Label: %i" % label)
    ax.set_axis_off()
plt.savefig(
    plot_dir / "01_scikit_digits_sample.png",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
plt.close()

# 2. Split data
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Train initial Random Forest model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_tr, y_tr)

score_tr = clf.score(X_tr, y_tr)
score_te = clf.score(X_te, y_te)
print(f"Initial RF Model accuracy on train data: {score_tr * 100:.2f}%")
print(f"Initial RF Model accuracy on test data:  {score_te * 100:.2f}%")

# 4. Fine-tune Random Forest model
parameters = {"max_depth": [5, 25, 50], "n_estimators": [1, 10, 50, 200]}

grid = GridSearchCV(RandomForestClassifier(random_state=42), parameters, cv=5)
grid.fit(X_tr, y_tr)

score_tr = grid.score(X_tr, y_tr)
score_te = grid.score(X_te, y_te)
print(f"\nTuned RF Model accuracy on train data: {score_tr * 100:.2f}%")
print(f"Tuned RF Model accuracy on test data:  {score_te * 100:.2f}%")

# Plot RF parameter heatmap
df_res = pd.DataFrame(grid.cv_results_)
df_res = df_res.iloc[:, df_res.columns.str.contains("mean_test_score|param_")]
df_res = df_res.astype("float")

result_table = df_res.pivot(
    index="param_max_depth", columns="param_n_estimators", values="mean_test_score"
)
plt.figure(figsize=(8, 6))
sns.heatmap(100 * result_table, annot=True, fmt=".2f", square=True, cbar=False)
plt.title("RF Accuracy on validation set, based on model hyper-parameter")
plt.savefig(
    plot_dir / "01_scikit_rf_heatmap.png",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
plt.close()

best_rf = grid.best_estimator_
print(f"\nBest RF parameters: {grid.best_params_}")

# 5. Train and tune SVM model
clf = SVC(kernel="rbf", random_state=42)
parameters = {
    "C": [0.01, 0.1, 1.0, 10.0, 100.0],
    "gamma": [0.00001, 0.0001, 0.001, 0.01, 0.1],
}

grid = GridSearchCV(clf, parameters, cv=5)
grid.fit(X_tr, y_tr)

score_tr = grid.score(X_tr, y_tr)
score_te = grid.score(X_te, y_te)
print(f"\nSVM Model accuracy on train data: {score_tr * 100:.2f}%")
print(f"SVM Model accuracy on test data:  {score_te * 100:.2f}%")

# Plot SVM parameter heatmap
df_res = pd.DataFrame(grid.cv_results_)
df_res = df_res.iloc[:, df_res.columns.str.contains("mean_test_score|param_")]
df_res = df_res.astype("float")

result_table = df_res.pivot(
    index="param_gamma", columns="param_C", values="mean_test_score"
)
plt.figure(figsize=(8, 6))
sns.heatmap(100 * result_table, annot=True, fmt=".2f", square=True)
plt.title("SVM Accuracy on validation set, based on model hyper-parameter")
plt.savefig(
    plot_dir / "01_scikit_svm_heatmap.png",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
plt.close()

best_svm = grid.best_estimator_
print(f"Best SVM parameters: {grid.best_params_}")

# 6. Post-model investigation
y_pred = best_svm.predict(X_te)
print("\nClassification Report:")
print(classification_report(y_te, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_te, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(pd.DataFrame(cm), annot=True, cbar=False, square=True)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.savefig(
    plot_dir / "01_scikit_confusion_matrix.png",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
plt.close()

# 7. Additional investigations
# Feature importance visualization
feat_import = best_rf.feature_importances_
feature_importance_image = feat_import.reshape(8, 8)
plt.figure(figsize=(5, 5))
plt.imshow(feature_importance_image)
plt.title("RF Feature Importance")
plt.savefig(
    plot_dir / "01_scikit_feature_importance.png",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
plt.close()

# Prediction probabilities visualization
y_prob = best_rf.predict_proba(X_te)
target_prob = [e[i] for e, i in zip(y_prob, y_te)]

# Most confident predictions
plt.figure(figsize=(9, 1.5))
_, axes = plt.subplots(nrows=3, ncols=20, figsize=(9, 1.5))
for ax, idx in zip(axes.ravel(), np.argsort(target_prob)[::-1]):
    ax.imshow(X_te[idx].reshape(8, 8), cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_axis_off()
plt.savefig(
    plot_dir / "01_scikit_confident_predictions.png",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
plt.close()

# Least confident predictions
plt.figure(figsize=(9, 1.5))
_, axes = plt.subplots(nrows=3, ncols=20, figsize=(9, 1.5))
for ax, idx in zip(axes.ravel(), np.argsort(target_prob)):
    ax.imshow(X_te[idx].reshape(8, 8), cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_axis_off()
plt.savefig(
    plot_dir / "01_scikit_uncertain_predictions.png",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
plt.close()
