import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Generate dummy dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                           n_classes=2, weights=[0.9, 0.1], random_state=42)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

# Store evaluation metrics
results = {}

# Train & Evaluate Models
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"--- {name} ---")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)

    results[name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc_score
    }

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC: {auc_score:.4f}")

# Plot ROC Curves
plt.figure(figsize=(8, 6))
for name, model in models.items():
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curves')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Feature Importances
def plot_feature_importance(model, model_name):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        features = [f"Feature {i}" for i in range(X.shape[1])]
        plt.figure(figsize=(8, 5))
        sns.barplot(x=importances[indices], y=np.array(features)[indices])
        plt.title(f"Feature Importances - {model_name}")
        plt.tight_layout()
        plt.show()

plot_feature_importance(models["Random Forest"], "Random Forest")
plot_feature_importance(models["XGBoost"], "XGBoost")

# Summary of all models
print("\nAccuracy Scores:")
for name, metrics in results.items():
    print(f"{name}: {metrics['accuracy']:.4f}")

# Correlation Matrix
df_corr = pd.DataFrame(X, columns=[f"Feature {i}" for i in range(X.shape[1])])
plt.figure(figsize=(10, 8))
sns.heatmap(df_corr.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()

# SHAP Summary Plot
try:
    explainer = shap.Explainer(models["XGBoost"], X_train)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, feature_names=[f"Feature {i}" for i in range(X.shape[1])])
except Exception as e:
    print(f"SHAP plotting skipped: {e}")
