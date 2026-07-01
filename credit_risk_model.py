#!/usr/bin/env python3
"""
Give Me Some Credit - 信用风险建模全流程
1. 数据加载与探索性分析 (EDA)
2. 数据预处理（缺失值、异常值处理）
3. 特征工程
4. 多模型训练与评估（逻辑回归、随机森林、XGBoost、LightGBM）
5. 生成可视化图表
6. 交叉验证
7. 对测试集进行违约概率预测
"""

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
os.makedirs('/tmp/matplotlib_cache', exist_ok=True)
os.environ['OMP_NUM_THREADS'] = '4'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                             confusion_matrix, brier_score_loss)
import xgboost as xgb
import lightgbm as lgb

# ============================================================
# 配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')

RANDOM_STATE = 42

# ============================================================
# 1. 数据加载
# ============================================================
print("=" * 60)
print("1. 数据加载")
print("=" * 60)

df_train = pd.read_csv(os.path.join(BASE_DIR, 'cs-training.csv'), index_col=0)
df_test = pd.read_csv(os.path.join(BASE_DIR, 'cs-test.csv'), index_col=0)

print(f"训练集形状: {df_train.shape}")
print(f"测试集形状: {df_test.shape}")
print(f"\n训练集字段: {list(df_train.columns)}")
print(f"\n目标变量分布:")
print(df_train['SeriousDlqin2yrs'].value_counts())
print(f"\n违约率: {df_train['SeriousDlqin2yrs'].mean():.4f} ({df_train['SeriousDlqin2yrs'].sum()}/{len(df_train)})")

# ============================================================
# 2. 探索性数据分析 (EDA)
# ============================================================
print("\n" + "=" * 60)
print("2. 探索性数据分析 (EDA)")
print("=" * 60)

print("\n--- 训练集基本信息 ---")
print(df_train.info())

print("\n--- 描述性统计 ---")
print(df_train.describe().round(3))

print("\n--- 缺失值统计 ---")
missing_train = df_train.isnull().sum()
missing_pct = (missing_train / len(df_train) * 100).round(2)
missing_df = pd.DataFrame({'缺失数量': missing_train, '缺失比例(%)': missing_pct})
print(missing_df[missing_df['缺失数量'] > 0])

# 目标变量分布图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax1 = df_train['SeriousDlqin2yrs'].value_counts().plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
axes[0].set_title('目标变量分布 (SeriousDlqin2yrs)', fontsize=13)
axes[0].set_xlabel('是否违约 (0=否, 1=是)')
axes[0].set_ylabel('数量')
for p in ax1.patches:
    axes[0].annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width()/2., p.get_height()),
                      ha='center', va='bottom', fontsize=11)

axes[1].pie(df_train['SeriousDlqin2yrs'].value_counts(), labels=['未违约', '违约'],
            autopct='%1.2f%%', colors=['#2ecc71', '#e74c3c'], startangle=90)
axes[1].set_title('违约率占比', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_target_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ 保存: 01_target_distribution.png")

# 特征分布
features = [c for c in df_train.columns if c != 'SeriousDlqin2yrs']
fig, axes = plt.subplots(4, 3, figsize=(18, 20))
for i, feat in enumerate(features):
    row, col = divmod(i, 3)
    data_no_na = df_train[feat].dropna()
    axes[row][col].hist(data_no_na, bins=50, color='#3498db', alpha=0.7, edgecolor='white')
    axes[row][col].set_title(f'{feat}', fontsize=11)
    axes[row][col].set_ylabel('频数')
plt.suptitle('特征分布直方图', fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_feature_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ 保存: 02_feature_distribution.png")

# 相关性热力图
fig, ax = plt.subplots(figsize=(12, 10))
corr = df_train.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
            center=0, square=True, linewidths=0.5, ax=ax, annot_kws={'size': 9})
ax.set_title('特征相关性热力图', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ 保存: 03_correlation_heatmap.png")

# 违约 vs 非违约 特征对比
fig, axes = plt.subplots(4, 3, figsize=(18, 20))
for i, feat in enumerate(features):
    row, col = divmod(i, 3)
    no_default = df_train[df_train['SeriousDlqin2yrs'] == 0][feat].dropna()
    default = df_train[df_train['SeriousDlqin2yrs'] == 1][feat].dropna()
    p1, p99 = no_default.quantile(0.01), no_default.quantile(0.99)
    axes[row][col].hist(no_default, bins=40, alpha=0.5, label='未违约', color='#2ecc71', density=True, range=(p1, p99))
    axes[row][col].hist(default, bins=40, alpha=0.5, label='违约', color='#e74c3c', density=True, range=(p1, p99))
    axes[row][col].set_title(feat, fontsize=11)
    axes[row][col].legend(fontsize=8)
plt.suptitle('违约 vs 非违约 特征分布对比', fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_default_vs_nodefault.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ 保存: 04_default_vs_nodefault.png")

# ============================================================
# 3. 数据预处理
# ============================================================
print("\n" + "=" * 60)
print("3. 数据预处理")
print("=" * 60)

def preprocess_data(df):
    """数据预处理：缺失值填充 + 异常值处理"""
    df = df.copy()

    # 3.1 缺失值处理
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(df['NumberOfDependents'].median())

    # 3.2 异常值处理
    df['age'] = df['age'].clip(lower=18, upper=100)
    for col in ['NumberOfTime30-59DaysPastDueNotWorse',
                'NumberOfTime60-89DaysPastDueNotWorse',
                'NumberOfTimes90DaysLate']:
        df[col] = df[col].apply(lambda x: min(x, 20))
    df['RevolvingUtilizationOfUnsecuredLines'] = df['RevolvingUtilizationOfUnsecuredLines'].clip(upper=10)
    df['DebtRatio'] = df['DebtRatio'].clip(upper=10)

    return df

df_train_processed = preprocess_data(df_train)
df_test_processed = preprocess_data(df_test)

print("缺失值处理: MonthlyIncome/NumberOfDependents 中位数填充")
print("异常值处理: age截断[18,100], 逾期次数截断max20, RevolvingUtilization/DebtRatio截断max10")
print(f"处理后缺失值: {df_train_processed.isnull().sum().sum()}")

# ============================================================
# 4. 特征工程
# ============================================================
print("\n" + "=" * 60)
print("4. 特征工程")
print("=" * 60)

def feature_engineering(df):
    df = df.copy()
    df['TotalPastDue'] = (df['NumberOfTime30-59DaysPastDueNotWorse'] +
                          df['NumberOfTime60-89DaysPastDueNotWorse'] +
                          df['NumberOfTimes90DaysLate'])
    df['HasPastDue'] = (df['TotalPastDue'] > 0).astype(int)
    df['SevereLateCount'] = df['NumberOfTimes90DaysLate']
    df['TotalCreditLines'] = (df['NumberOfOpenCreditLinesAndLoans'] +
                               df['NumberRealEstateLoansOrLines'])
    df['RealEstateRatio'] = df['NumberRealEstateLoansOrLines'] / (df['TotalCreditLines'] + 1)
    df['IncomePerDependent'] = df['MonthlyIncome'] / (df['NumberOfDependents'] + 1)
    df['LogMonthlyIncome'] = np.log1p(df['MonthlyIncome'])
    return df

df_train_fe = feature_engineering(df_train_processed)
df_test_fe = feature_engineering(df_test_processed)

new_features = ['TotalPastDue', 'HasPastDue', 'SevereLateCount',
                'TotalCreditLines', 'RealEstateRatio', 'IncomePerDependent', 'LogMonthlyIncome']
print(f"新增衍生特征 ({len(new_features)}个): {new_features}")
feature_cols = [c for c in df_train_fe.columns if c != 'SeriousDlqin2yrs']
print(f"最终特征数量: {len(feature_cols)}")

# ============================================================
# 5. 模型训练与评估
# ============================================================
print("\n" + "=" * 60)
print("5. 模型训练与评估")
print("=" * 60)

X = df_train_fe[feature_cols]
y = df_train_fe['SeriousDlqin2yrs']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
print(f"训练集: {X_train.shape[0]} | 验证集: {X_val.shape[0]}")

models = {}

# 5.1 逻辑回归
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)
lr.fit(X_train_scaled, y_train)
lr_pred_val = lr.predict_proba(X_val_scaled)[:, 1]
lr_auc = roc_auc_score(y_val, lr_pred_val)
models['Logistic Regression'] = {'model': lr, 'scaler': scaler, 'auc': lr_auc, 'y_pred_val': lr_pred_val, 'y_val': y_val}
print(f"逻辑回归          — AUC: {lr_auc:.4f}")

# 5.2 随机森林
rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=50,
                            class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred_val = rf.predict_proba(X_val)[:, 1]
rf_auc = roc_auc_score(y_val, rf_pred_val)
models['Random Forest'] = {'model': rf, 'scaler': None, 'auc': rf_auc, 'y_pred_val': rf_pred_val, 'y_val': y_val}
print(f"随机森林          — AUC: {rf_auc:.4f}")

# 5.3 XGBoost
scale_pos = len(y_train[y_train==0]) / len(y_train[y_train==1])
xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    random_state=RANDOM_STATE, eval_metric='auc', n_jobs=4,
    early_stopping_rounds=20
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
xgb_pred_val = xgb_model.predict_proba(X_val)[:, 1]
xgb_auc = roc_auc_score(y_val, xgb_pred_val)
models['XGBoost'] = {'model': xgb_model, 'scaler': None, 'auc': xgb_auc, 'y_pred_val': xgb_pred_val, 'y_val': y_val}
print(f"XGBoost           — AUC: {xgb_auc:.4f}")

# 5.4 LightGBM
lgb_model = lgb.LGBMClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    class_weight='balanced',
    random_state=RANDOM_STATE, n_jobs=4, verbose=-1
)
lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(20, verbose=False)])
lgb_pred_val = lgb_model.predict_proba(X_val)[:, 1]
lgb_auc = roc_auc_score(y_val, lgb_pred_val)
models['LightGBM'] = {'model': lgb_model, 'scaler': None, 'auc': lgb_auc, 'y_pred_val': lgb_pred_val, 'y_val': y_val}
print(f"LightGBM          — AUC: {lgb_auc:.4f}")

# ============================================================
# 6. 模型对比与评估
# ============================================================
print("\n" + "=" * 60)
print("6. 模型对比与评估")
print("=" * 60)

def calculate_ks(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return max(tpr - fpr)

results = []
for name, info in models.items():
    y_v = info['y_val']
    y_p = info['y_pred_val']
    auc = info['auc']
    ks = calculate_ks(y_v, y_p)
    brier = brier_score_loss(y_v, y_p)
    prec, rec, thresh = precision_recall_curve(y_v, y_p)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-10)
    best_f1 = f1_scores.max()
    best_thresh = thresh[f1_scores.argmax()] if len(thresh) > 1 else 0.5
    results.append({
        '模型': name, 'AUC': round(auc, 4), 'KS': round(ks, 4),
        'Brier Score': round(brier, 4), 'Best F1': round(best_f1, 4),
        'Best Threshold': round(best_thresh, 4)
    })

results_df = pd.DataFrame(results).sort_values('AUC', ascending=False)
print("\n模型评估汇总:")
print(results_df.to_string(index=False))
results_df.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison.csv'), index=False)

best_model_name = results_df.iloc[0]['模型']
print(f"\n>>> 最佳模型: {best_model_name} (AUC={results_df.iloc[0]['AUC']})")

# ROC曲线对比
fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
for i, (name, info) in enumerate(models.items()):
    fpr, tpr, _ = roc_curve(info['y_val'], info['y_pred_val'])
    ax.plot(fpr, tpr, label=f'{name} (AUC={info["auc"]:.4f})', color=colors[i], linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.set_xlabel('假正率 (FPR)', fontsize=12)
ax.set_ylabel('真正率 (TPR)', fontsize=12)
ax.set_title('ROC 曲线对比', fontsize=14)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05_roc_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ 保存: 05_roc_curves.png")

# PR曲线对比
fig, ax = plt.subplots(figsize=(10, 8))
for i, (name, info) in enumerate(models.items()):
    prec, rec, _ = precision_recall_curve(info['y_val'], info['y_pred_val'])
    ax.plot(rec, prec, label=name, color=colors[i], linewidth=2)
ax.set_xlabel('召回率 (Recall)', fontsize=12)
ax.set_ylabel('精确率 (Precision)', fontsize=12)
ax.set_title('Precision-Recall 曲线对比', fontsize=14)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '06_pr_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ 保存: 06_pr_curves.png")

# KS曲线（最佳模型）
best_info = models[best_model_name]
fig, ax = plt.subplots(figsize=(10, 8))
fpr, tpr, thresholds = roc_curve(best_info['y_val'], best_info['y_pred_val'])
ks_value = max(tpr - fpr)
ks_idx = np.argmax(tpr - fpr)
ax.plot(thresholds, tpr, label='TPR (真正率)', color='#e74c3c', linewidth=2)
ax.plot(thresholds, fpr, label='FPR (假正率)', color='#3498db', linewidth=2)
ax.plot(thresholds, tpr - fpr, label=f'KS = {ks_value:.4f}', color='#2ecc71', linewidth=2, linestyle='--')
ax.axvline(x=thresholds[ks_idx], color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('阈值', fontsize=12)
ax.set_ylabel('比率', fontsize=12)
ax.set_title(f'KS 曲线 — {best_model_name}', fontsize=14)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '07_ks_curve.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ 保存: 07_ks_curve.png")

# 特征重要性（树模型）
tree_models = {k: v for k, v in models.items() if v['scaler'] is None and k != 'Logistic Regression'}
n_tree = min(3, len(tree_models))
fig, axes = plt.subplots(1, n_tree, figsize=(7 * n_tree, 8))
if n_tree == 1:
    axes = [axes]
for i, (name, info) in enumerate(tree_models.items()):
    model = info['model']
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feat_df = pd.DataFrame({'feature': feature_cols, 'importance': importances})
        feat_df = feat_df.sort_values('importance', ascending=True).tail(15)
        axes[i].barh(feat_df['feature'], feat_df['importance'], color='#3498db')
        axes[i].set_title(f'{name} — 特征重要性 Top15', fontsize=12)
        axes[i].set_xlabel('重要性')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '08_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ 保存: 08_feature_importance.png")

# 混淆矩阵
best_thresh_val = results_df[results_df['模型'] == best_model_name]['Best Threshold'].values[0]
best_pred_binary = (best_info['y_pred_val'] >= best_thresh_val).astype(int)
cm = confusion_matrix(best_info['y_val'], best_pred_binary)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['未违约', '违约'], yticklabels=['未违约', '违约'])
ax.set_title(f'混淆矩阵 — {best_model_name} (阈值={best_thresh_val:.3f})', fontsize=13)
ax.set_xlabel('预测值', fontsize=12)
ax.set_ylabel('真实值', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '09_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ 保存: 09_confusion_matrix.png")

# ============================================================
# 7. 交叉验证（轻量化，3折）
# ============================================================
print("\n" + "=" * 60)
print("7. 交叉验证 (3-fold)")
print("=" * 60)

best_model = best_info['model']
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

# 轻量化CV模型以避免内存溢出
if best_model_name == 'XGBoost':
    cv_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, scale_pos_weight=scale_pos,
        random_state=RANDOM_STATE, n_jobs=4, eval_metric='auc'
    )
elif best_model_name == 'LightGBM':
    cv_model = lgb.LGBMClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        subsample=0.8, class_weight='balanced',
        random_state=RANDOM_STATE, n_jobs=4, verbose=-1
    )
elif best_model_name == 'Logistic Regression':
    cv_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)
    X = pd.DataFrame(scaler.transform(X), columns=feature_cols)
else:
    cv_model = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=100,
                                      class_weight='balanced', random_state=RANDOM_STATE, n_jobs=4)

cv_scores = cross_val_score(cv_model, X, y, cv=cv, scoring='roc_auc')
print(f"最佳模型 ({best_model_name}) 3折交叉验证 AUC:")
print(f"  各折AUC: {[f'{s:.4f}' for s in cv_scores]}")
print(f"  平均AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# ============================================================
# 8. 测试集预测
# ============================================================
print("\n" + "=" * 60)
print("8. 测试集预测")
print("=" * 60)

X_test = df_test_fe[feature_cols]

if best_model_name == 'Logistic Regression':
    X_test_scaled = scaler.transform(X_test)
    test_pred = best_model.predict_proba(X_test_scaled)[:, 1]
else:
    test_pred = best_model.predict_proba(X_test)[:, 1]

test_result = pd.DataFrame({
    'Id': df_test.index,
    'Probability': test_pred
})
test_result.to_csv(os.path.join(BASE_DIR, 'prediction_results.csv'), index=False)
print(f"预测结果已保存: prediction_results.csv")
print(f"预测样本数: {len(test_result)}")
print(f"\n预测概率统计:")
print(test_result['Probability'].describe().round(4))

# 预测概率分布
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(test_pred, bins=50, color='#9b59b6', alpha=0.7, edgecolor='white')
ax.axvline(x=test_pred.mean(), color='#e74c3c', linestyle='--', linewidth=2, label=f'均值={test_pred.mean():.4f}')
ax.set_xlabel('预测违约概率', fontsize=12)
ax.set_ylabel('样本数', fontsize=12)
ax.set_title('测试集预测违约概率分布', fontsize=14)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '10_prediction_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ 保存: 10_prediction_distribution.png")

# ============================================================
# 9. 保存模型与汇总
# ============================================================
import joblib
model_package = {
    'model': best_model,
    'scaler': scaler if best_model_name == 'Logistic Regression' else None,
    'feature_cols': feature_cols,
    'best_threshold': best_thresh_val,
    'model_name': best_model_name,
    'cv_auc_mean': cv_scores.mean(),
    'cv_auc_std': cv_scores.std(),
    'val_auc': best_info['auc'],
    'val_ks': calculate_ks(best_info['y_val'], best_info['y_pred_val'])
}
joblib.dump(model_package, os.path.join(OUTPUT_DIR, 'best_model.pkl'))
print(f"\n模型已保存: output/best_model.pkl")

# 保存评估结果汇总
summary = {
    '数据集': 'Give Me Some Credit',
    '训练集样本数': len(df_train),
    '测试集样本数': len(df_test),
    '违约率': f"{df_train['SeriousDlqin2yrs'].mean():.4f}",
    '特征数(原始)': 10,
    '特征数(衍生后)': len(feature_cols),
    '最佳模型': best_model_name,
    '验证集AUC': round(best_info['auc'], 4),
    '验证集KS': round(calculate_ks(best_info['y_val'], best_info['y_pred_val']), 4),
    '交叉验证AUC': f"{cv_scores.mean():.4f} +/- {cv_scores.std():.4f}",
    '最优阈值': round(best_thresh_val, 4),
    'Brier Score': round(brier_score_loss(best_info['y_val'], best_info['y_pred_val']), 4)
}
summary_df = pd.DataFrame(list(summary.items()), columns=['指标', '值'])
summary_df.to_csv(os.path.join(OUTPUT_DIR, 'model_summary.csv'), index=False)

print("\n" + "=" * 60)
print("全流程完成!")
print("=" * 60)
print(f"\n所有输出文件位于: output/")
print(f"预测结果: prediction_results.csv")
