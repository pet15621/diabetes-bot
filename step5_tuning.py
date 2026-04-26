import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)

plt.rcParams['font.family'] = 'Tahoma'
# ========== โหลดข้อมูล ==========
df = pd.read_csv("diabetes.csv")
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zero:
    df[col] = df[col].replace(0, df[col].median())

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler     = joblib.load("models/scaler.pkl")
X_train_sc = scaler.transform(X_train)
X_test_sc  = scaler.transform(X_test)

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ========================================
# Helper — วัด metrics
# ========================================
def get_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall":    round(recall_score(y_test, y_pred), 4),
        "F1":        round(f1_score(y_test, y_pred), 4),
        "ROC-AUC":   round(roc_auc_score(y_test, y_prob), 4),
    }

# ========================================
# 1. Logistic Regression
# ========================================
print("=" * 50)
print("Logistic Regression — Tuning")
print("=" * 50)

# Before
lr_before = LogisticRegression(max_iter=1000, random_state=42)
lr_before.fit(X_train_sc, y_train)
m_lr_before = get_metrics(lr_before, X_test_sc, y_test)
print(f"Before: {m_lr_before}")

# GridSearch
lr_params = {
    "C":       [0.01, 0.1, 1, 10, 100],
    "solver":  ["lbfgs", "saga"],
    "penalty": ["l2"]
}
lr_grid = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    lr_params, cv=5, scoring="roc_auc", n_jobs=-1)
lr_grid.fit(X_train_sc, y_train)

print(f"Best params: {lr_grid.best_params_}")
lr_after = lr_grid.best_estimator_
m_lr_after = get_metrics(lr_after, X_test_sc, y_test)
print(f"After:  {m_lr_after}")

# ========================================
# 2. Decision Tree
# ========================================
print("\n" + "=" * 50)
print("Decision Tree — Tuning")
print("=" * 50)

# Before
dt_before = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_before.fit(X_train_sc, y_train)
m_dt_before = get_metrics(dt_before, X_test_sc, y_test)
print(f"Before: {m_dt_before}")

# GridSearch
dt_params = {
    "max_depth":        [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10],
    "criterion":        ["gini", "entropy"]
}
dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_params, cv=5, scoring="roc_auc", n_jobs=-1)
dt_grid.fit(X_train_sc, y_train)

print(f"Best params: {dt_grid.best_params_}")
dt_after = dt_grid.best_estimator_
m_dt_after = get_metrics(dt_after, X_test_sc, y_test)
print(f"After:  {m_dt_after}")

# ========================================
# 3. ANN (MLP)
# ========================================
print("\n" + "=" * 50)
print("ANN (MLP) — Tuning")
print("=" * 50)

# Before
ann_before = MLPClassifier(
    hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
ann_before.fit(X_train_sc, y_train)
m_ann_before = get_metrics(ann_before, X_test_sc, y_test)
print(f"Before: {m_ann_before}")

# GridSearch (ลด combinations ให้ไม่นานเกิน)
ann_params = {
    "hidden_layer_sizes": [(64, 32), (128, 64), (128, 64, 32), (32,)],
    "activation":         ["relu", "tanh"],
    "learning_rate_init": [0.001, 0.01],
    "max_iter":           [1000]
}
ann_grid = GridSearchCV(
    MLPClassifier(random_state=42),
    ann_params, cv=5, scoring="roc_auc", n_jobs=-1,
    verbose=1)
ann_grid.fit(X_train_sc, y_train)

print(f"Best params: {ann_grid.best_params_}")
ann_after = ann_grid.best_estimator_
m_ann_after = get_metrics(ann_after, X_test_sc, y_test)
print(f"After:  {m_ann_after}")

# ========================================
# สรุปตาราง Before vs After
# ========================================
print("\n" + "=" * 50)
print("สรุปผล Before vs After Tuning")
print("=" * 50)

summary = {
    "LR Before":  m_lr_before,
    "LR After":   m_lr_after,
    "DT Before":  m_dt_before,
    "DT After":   m_dt_after,
    "ANN Before": m_ann_before,
    "ANN After":  m_ann_after,
}
df_summary = pd.DataFrame(summary).T
print(df_summary.to_string())

# ========================================
# กราฟ Before vs After
# ========================================
metrics   = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
models    = ["Logistic Regression", "Decision Tree", "ANN (MLP)"]
befores   = [m_lr_before, m_dt_before, m_ann_before]
afters    = [m_lr_after,  m_dt_after,  m_ann_after]
colors_b  = ["#AED6F1", "#A9DFBF", "#F9E79F"]
colors_a  = ["#2E86C1", "#1E8449", "#D4AC0D"]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Before vs After Tuning — เปรียบเทียบ 3 โมเดล",
             fontsize=14, fontweight='bold')

for ax, model, before, after, cb, ca in zip(
        axes, models, befores, afters, colors_b, colors_a):
    x     = np.arange(len(metrics))
    width = 0.35
    bars1 = ax.bar(x - width/2,
                   [before[m] for m in metrics],
                   width, label="Before", color=cb, edgecolor='gray')
    bars2 = ax.bar(x + width/2,
                   [after[m] for m in metrics],
                   width, label="After",  color=ca, edgecolor='gray')

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f'{bar.get_height():.2f}',
                ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f'{bar.get_height():.2f}',
                ha='center', va='bottom', fontsize=7)

    ax.set_title(model, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/10_tuning_comparison.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ บันทึก 10_tuning_comparison.png")

# ========================================
# กราฟ ROC Curve Before vs After
# ========================================
from sklearn.metrics import roc_curve, auc

pairs = [
    ("Logistic Regression", lr_before, lr_after),
    ("Decision Tree",       dt_before, dt_after),
    ("ANN (MLP)",           ann_before, ann_after),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("ROC Curve Before vs After Tuning", fontsize=14, fontweight='bold')

for ax, (name, before, after) in zip(axes, pairs):
    for model, label, color, ls in [
        (before, "Before", "#AED6F1", "--"),
        (after,  "After",  "#2E86C1", "-"),
    ]:
        y_prob = model.predict_proba(X_test_sc)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, ls=ls, lw=2,
                label=f"{label} (AUC={roc_auc:.3f})")
    ax.plot([0,1],[0,1],'k--', lw=1)
    ax.set_title(name, fontweight='bold')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/11_roc_before_after.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ บันทึก 11_roc_before_after.png")

# ========================================
# กราฟ Radar Chart Before vs After
# ========================================
categories = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True))
fig.suptitle("Radar Chart Before vs After Tuning", fontsize=14, fontweight='bold')

pairs_metrics = [
    ("Logistic Regression", m_lr_before, m_lr_after),
    ("Decision Tree",       m_dt_before, m_dt_after),
    ("ANN (MLP)",           m_ann_before, m_ann_after),
]
for ax, (name, before, after) in zip(axes, pairs_metrics):
    vals_b = [before[m] for m in categories] + [before[categories[0]]]
    vals_a = [after[m]  for m in categories] + [after[categories[0]]]
    ax.plot(angles, vals_b, 'o--', color="#AED6F1", lw=2, label="Before")
    ax.fill(angles, vals_b, color="#AED6F1", alpha=0.2)
    ax.plot(angles, vals_a, 'o-',  color="#2E86C1", lw=2, label="After")
    ax.fill(angles, vals_a, color="#2E86C1", alpha=0.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title(name, fontweight='bold', pad=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.tight_layout()
plt.savefig("outputs/12_radar_before_after.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ บันทึก 12_radar_before_after.png")

# ========================================
# Save โมเดลที่ tuned แล้ว แยก folder
# ========================================
os.makedirs("models/tuned", exist_ok=True)

joblib.dump(lr_after,  "models/tuned/logistic.pkl")
joblib.dump(dt_after,  "models/tuned/decision_tree.pkl")
joblib.dump(ann_after, "models/tuned/ann.pkl")
print("✅ Save tuned models ไว้ที่ models/tuned/")
print("\n🎉 Tuning เสร็จครบแล้ว!")