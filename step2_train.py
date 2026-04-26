import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report)
import joblib
import os

# ========== 1. โหลดข้อมูล ==========
df = pd.read_csv("diabetes.csv")
print("Shape:", df.shape)
print(df.head())

# ========== 2. Preprocessing ==========
# แทนค่า 0 ที่เป็นไปไม่ได้ด้วย Median
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zero:
    df[col] = df[col].replace(0, df[col].median())

# แบ่ง features และ target
SELECTED_FEATURES = ['Glucose', 'BMI', 'Age', 'Pregnancies']
X = df[SELECTED_FEATURES]
y = df['Outcome']

# แบ่ง train/test 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale ข้อมูล
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ========== 3. Train 3 Models ==========
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
    "ANN (MLP)":           MLPClassifier(hidden_layer_sizes=(64, 32),
                                         max_iter=500, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]

    results[name] = {
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall":    round(recall_score(y_test, y_pred), 4),
        "F1":        round(f1_score(y_test, y_pred), 4),
        "ROC-AUC":   round(roc_auc_score(y_test, y_prob), 4),
    }
    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred))

# ========== 4. แสดงตารางเปรียบเทียบ ==========
print("\n====== Model Comparison ======")
print(pd.DataFrame(results).T.to_string())

# ========== 5. Save Model ที่ดีที่สุด + Scaler ==========
os.makedirs("models", exist_ok=True)

# Save ทุกตัวไว้เลย
joblib.dump(models["Logistic Regression"], "models/logistic.pkl")
joblib.dump(models["Decision Tree"],       "models/decision_tree.pkl")
joblib.dump(models["ANN (MLP)"],           "models/ann.pkl")
joblib.dump(scaler,                        "models/scaler.pkl")
joblib.dump(list(X.columns),              "models/feature_names.pkl")

print("\n✅ บันทึก models เรียบร้อยแล้ว!")