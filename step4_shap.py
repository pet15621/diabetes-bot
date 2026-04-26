import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import os
from sklearn.model_selection import train_test_split

plt.rcParams['font.family'] = 'Tahoma'
# ========== โหลดข้อมูล + Preprocess ==========
df = pd.read_csv("diabetes.csv")

cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zero:
    df[col] = df[col].replace(0, df[col].median())

SELECTED_FEATURES = ['Glucose', 'BMI', 'Age', 'Pregnancies']
X = df[SELECTED_FEATURES]
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler      = joblib.load("models/scaler.pkl")
X_train_sc  = scaler.transform(X_train)
X_test_sc   = scaler.transform(X_test)
X_test_df   = pd.DataFrame(X_test_sc, columns=X.columns)
X_train_df  = pd.DataFrame(X_train_sc, columns=X.columns)

lr  = joblib.load("models/logistic.pkl")
dt  = joblib.load("models/decision_tree.pkl")
ann = joblib.load("models/ann.pkl")

os.makedirs("outputs", exist_ok=True)

# ========================================
# SHAP — Logistic Regression
# ========================================
print("กำลังคำนวณ SHAP — Logistic Regression...")
explainer_lr    = shap.LinearExplainer(lr, X_train_df)
shap_values_lr  = explainer_lr.shap_values(X_test_df)

plt.figure()
shap.summary_plot(shap_values_lr, X_test_df,
                  plot_type="bar", show=False)
plt.title("SHAP Feature Importance — Logistic Regression",
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/07_shap_logistic.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ 07_shap_logistic.png")

# ========================================
# SHAP — Decision Tree
# ========================================
print("กำลังคำนวณ SHAP — Decision Tree...")
explainer_dt   = shap.TreeExplainer(dt)
shap_values_dt = explainer_dt.shap_values(X_test_df)

sv_dt = shap_values_dt[1] if isinstance(shap_values_dt, list) else shap_values_dt

plt.figure()
shap.summary_plot(sv_dt, X_test_df,
                  plot_type="bar", show=False)
plt.title("SHAP Feature Importance — Decision Tree",
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/08_shap_decision_tree.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ 08_shap_decision_tree.png")

# ========================================
# SHAP — ANN
# ========================================
print("กำลังคำนวณ SHAP — ANN... (รอสัก 1-2 นาที)")
background      = shap.sample(X_train_df, 50, random_state=42)
explainer_ann   = shap.KernelExplainer(ann.predict_proba, background)
shap_values_ann = explainer_ann.shap_values(
                      X_test_df.iloc[:50], nsamples=100)

sv_ann = shap_values_ann[1] if isinstance(shap_values_ann, list) else shap_values_ann

plt.figure()
shap.summary_plot(sv_ann, X_test_df.iloc[:50],
                  plot_type="bar", show=False)
plt.title("SHAP Feature Importance — ANN (MLP)",
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/09_shap_ann.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ 09_shap_ann.png")

# ========================================
# Save explainers ไว้ใช้ใน Line Bot
# ========================================

# lr และ dt save ได้ปกติ
joblib.dump(explainer_lr, "models/explainer_lr.pkl")
joblib.dump(explainer_dt, "models/explainer_dt.pkl")

# ann — KernelExplainer save ไม่ได้
# save background ไว้แทน แล้วสร้าง explainer ใหม่ตอนใช้งานจริง
joblib.dump(background, "models/shap_background.pkl")

print("\n✅ Save explainers เรียบร้อย!")

# ========================================
# ทดสอบ — อธิบายผลคนที่ 0 (ตัวอย่างจริง)
# ========================================
print("\n====== ทดสอบอธิบายผลรายบุคคล ======")
sample     = X_test_df.iloc[[0]]
sample_raw = X_test.iloc[0]
prob       = ann.predict_proba(sample)[0][1]

# สร้าง explainer ใหม่จาก background ที่ save ไว้
bg           = joblib.load("models/shap_background.pkl")
exp_ann_test = shap.KernelExplainer(ann.predict_proba, bg)
sv_single    = exp_ann_test.shap_values(sample, nsamples=100)
sv_single1 = np.array(sv_single).flatten()[4:]

feature_impact = pd.Series(np.abs(sv_single1), index=X.columns)
top3 = feature_impact.nlargest(3)

print(f"ความเสี่ยงเบาหวาน: {prob*100:.1f}%")
print("\nTop 3 สาเหตุหลัก:")
for feat, val in top3.items():
    actual = sample_raw[feat]
    print(f"  - {feat}: {actual:.1f}  (impact: {val:.3f})")

print("\n🎉 SHAP เสร็จครบแล้ว! ดูกราฟได้ที่ outputs/")