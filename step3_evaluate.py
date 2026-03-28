import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, roc_curve, 
                             auc, ConfusionMatrixDisplay)
import joblib
import os
plt.rcParams['font.family'] = 'Tahoma'
# ========== โหลดข้อมูลและ preprocess เหมือนเดิม ==========
df = pd.read_csv("diabetes.csv")
df_raw = df.copy()  # เก็บข้อมูลดิบไว้ดู outlier ก่อน clean

cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zero:
    df[col] = df[col].replace(0, df[col].median())

X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
scaler = joblib.load("models/scaler.pkl")
X_test_sc = scaler.transform(X_test)

# โหลด 3 โมเดล
lr  = joblib.load("models/logistic.pkl")
dt  = joblib.load("models/decision_tree.pkl")
ann = joblib.load("models/ann.pkl")

os.makedirs("outputs", exist_ok=True)

# ====================================================
# กราฟที่ 1 — Boxplot ทุก Feature (ดู Outlier)
# ====================================================
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle("Boxplot — Outlier Detection (ข้อมูลดิบก่อน Clean)", 
             fontsize=14, fontweight='bold')
features = df_raw.drop('Outcome', axis=1).columns
for i, col in enumerate(features):
    ax = axes[i // 4][i % 4]
    ax.boxplot(df_raw[col].dropna(), patch_artist=True,
               boxprops=dict(facecolor='#AED6F1', color='#2E86C1'),
               medianprops=dict(color='red', linewidth=2),
               flierprops=dict(marker='o', color='red', alpha=0.5))
    ax.set_title(col, fontsize=11)
    ax.set_ylabel("Value")
    # แสดง outlier count
    Q1 = df_raw[col].quantile(0.25)
    Q3 = df_raw[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df_raw[(df_raw[col] < Q1 - 1.5*IQR) | 
                      (df_raw[col] > Q3 + 1.5*IQR)][col]
    ax.set_xlabel(f"Outliers: {len(outliers)} จุด", 
                  fontsize=9, color='red')
plt.tight_layout()
plt.savefig("outputs/01_boxplot_outliers.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ บันทึก 01_boxplot_outliers.png")

# ====================================================
# กราฟที่ 2 — Distribution (Histogram) ทุก Feature
# ====================================================
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle("Distribution of Features", fontsize=14, fontweight='bold')
for i, col in enumerate(features):
    ax = axes[i // 4][i % 4]
    ax.hist(df[col], bins=20, color='#82E0AA', edgecolor='#1E8449')
    ax.set_title(col)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/02_distribution.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ บันทึก 02_distribution.png")

# ====================================================
# กราฟที่ 3 — Correlation Heatmap
# ====================================================
plt.figure(figsize=(10, 8))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            mask=mask, vmin=-1, vmax=1,
            linewidths=0.5, square=True)
plt.title("Correlation Heatmap", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/03_correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ บันทึก 03_correlation_heatmap.png")

# ====================================================
# กราฟที่ 4 — Confusion Matrix ทั้ง 3 โมเดล
# ====================================================
model_dict = {
    "Logistic Regression": lr,
    "Decision Tree": dt,
    "ANN (MLP)": ann
}
colors = ["Blues", "Greens", "Oranges"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Confusion Matrix — เปรียบเทียบ 3 โมเดล", 
             fontsize=14, fontweight='bold')
for ax, (name, model), cmap in zip(axes, model_dict.items(), colors):
    y_pred = model.predict(X_test_sc)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["ไม่เป็นเบาหวาน", "เป็นเบาหวาน"])
    disp.plot(ax=ax, colorbar=False, cmap=cmap)
    ax.set_title(name, fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/04_confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ บันทึก 04_confusion_matrix.png")

# ====================================================
# กราฟที่ 5 — ROC Curve ทั้ง 3 โมเดล
# ====================================================
plt.figure(figsize=(8, 6))
line_styles = ['-', '--', '-.']
line_colors = ['#2E86C1', '#1E8449', '#D35400']

for (name, model), ls, lc in zip(model_dict.items(), line_styles, line_colors):
    y_prob = model.predict_proba(X_test_sc)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, linestyle=ls, color=lc, linewidth=2,
             label=f"{name} (AUC = {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label="Random Guess")
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curve — เปรียบเทียบ 3 โมเดล", fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/05_roc_curve.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ บันทึก 05_roc_curve.png")

# ====================================================
# กราฟที่ 6 — Bar Chart เปรียบเทียบ Metrics
# ====================================================
metrics_data = {}
for name, model in model_dict.items():
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score, roc_auc_score)
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]
    metrics_data[name] = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "F1-Score":  f1_score(y_test, y_pred),
        "ROC-AUC":  roc_auc_score(y_test, y_prob),
    }

metrics_df = pd.DataFrame(metrics_data).T
x = np.arange(len(metrics_df.columns))
width = 0.25
bar_colors = ['#2E86C1', '#1E8449', '#D35400']

fig, ax = plt.subplots(figsize=(12, 6))
for i, (name, color) in enumerate(zip(metrics_df.index, bar_colors)):
    bars = ax.bar(x + i*width, metrics_df.loc[name], width,
                  label=name, color=color, alpha=0.85)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f'{bar.get_height():.2f}',
                ha='center', va='bottom', fontsize=8)

ax.set_xticks(x + width)
ax.set_xticklabels(metrics_df.columns, fontsize=11)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Performance Comparison", fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/06_model_comparison.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ บันทึก 06_model_comparison.png")

print("\n🎉 สร้างกราฟครบทั้งหมดแล้ว! ดูได้ที่โฟลเดอร์ outputs/")

# ========== Boxplot หลัง Clean ==========
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle("Boxplot — หลัง Clean (แทนค่า 0 ด้วย Median)", 
             fontsize=14, fontweight='bold')
for i, col in enumerate(features):
    ax = axes[i // 4][i % 4]
    ax.boxplot(df[col].dropna(), patch_artist=True,
               boxprops=dict(facecolor='#A9DFBF', color='#1E8449'),
               medianprops=dict(color='red', linewidth=2),
               flierprops=dict(marker='o', color='orange', alpha=0.5))
    ax.set_title(col, fontsize=11)
    ax.set_ylabel("Value")
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | 
                  (df[col] > Q3 + 1.5*IQR)][col]
    ax.set_xlabel(f"Outliers: {len(outliers)} จุด", 
                  fontsize=9, color='darkorange')
plt.tight_layout()
plt.savefig("outputs/01b_boxplot_after_clean.png", 
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ บันทึก 01b_boxplot_after_clean.png")