#!/usr/bin/env python3
"""生成HTML综合报告"""
import os, base64, pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

def img_to_base64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

# 读取模型对比和汇总
model_comp = pd.read_csv(os.path.join(OUTPUT_DIR, 'model_comparison.csv'))
summary = pd.read_csv(os.path.join(OUTPUT_DIR, 'model_summary.csv'))
pred = pd.read_csv(os.path.join(BASE_DIR, 'prediction_results.csv'))

# 图片
images = {}
for fname in sorted(os.listdir(OUTPUT_DIR)):
    if fname.endswith('.png'):
        images[fname] = img_to_base64(os.path.join(OUTPUT_DIR, fname))

# 构建模型对比表
model_table = model_comp.to_html(index=False, classes='data-table', border=0)

# 构建汇总
summary_dict = dict(zip(summary['指标'], summary['值']))
summary_rows = ""
for k, v in summary_dict.items():
    summary_rows += f"<tr><td class='label'>{k}</td><td class='value'>{v}</td></tr>"

# 图片HTML
def img_html(fname, title):
    return f"""
    <div class="chart-card">
        <h3>{title}</h3>
        <img src="data:image/png;base64,{images[fname]}" alt="{title}">
    </div>
    """

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>信用风险模型报告 — Give Me Some Credit</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
            background: #f0f2f5; color: #333; line-height: 1.8; }}
    .container {{ max-width: 1100px; margin: 0 auto; padding: 20px; }}

    /* 封面 */
    .cover {{
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: #fff; padding: 60px 40px; border-radius: 16px;
        text-align: center; margin-bottom: 30px; box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }}
    .cover h1 {{ font-size: 2.2em; margin-bottom: 10px; letter-spacing: 2px; }}
    .cover .subtitle {{ font-size: 1.1em; opacity: 0.8; }}
    .cover .date {{ font-size: 0.9em; opacity: 0.6; margin-top: 15px; }}

    /* 区块标题 */
    .section {{ background: #fff; border-radius: 12px; padding: 30px; margin-bottom: 24px;
                box-shadow: 0 2px 12px rgba(0,0,0,0.06); }}
    .section h2 {{ font-size: 1.5em; color: #1a1a2e; margin-bottom: 20px;
                   padding-bottom: 10px; border-bottom: 3px solid #0f3460; }}
    .section h3 {{ font-size: 1.15em; color: #333; margin: 15px 0 10px; }}

    /* 表格 */
    .data-table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 0.95em; }}
    .data-table th {{ background: #1a1a2e; color: #fff; padding: 12px 15px; text-align: left; font-weight: 600; }}
    .data-table td {{ padding: 10px 15px; border-bottom: 1px solid #eee; }}
    .data-table tr:nth-child(even) {{ background: #f8f9fa; }}
    .data-table tr:hover {{ background: #e8f0fe; }}

    /* 汇总卡片 */
    .summary-table {{ width: 100%; border-collapse: collapse; }}
    .summary-table td {{ padding: 10px 15px; border-bottom: 1px solid #eee; }}
    .summary-table td.label {{ font-weight: 600; color: #555; width: 40%; background: #f8f9fa; }}
    .summary-table td.value {{ color: #1a1a2e; font-weight: 500; }}

    /* 图片 */
    .chart-card {{ margin: 20px 0; text-align: center; }}
    .chart-card img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
    .chart-card h3 {{ text-align: left; }}

    /* 指标高亮 */
    .metrics {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 20px 0; }}
    .metric-card {{ flex: 1; min-width: 180px; background: linear-gradient(135deg, #0f3460, #1a1a2e);
                    color: #fff; padding: 20px; border-radius: 10px; text-align: center; }}
    .metric-card .num {{ font-size: 2em; font-weight: 700; }}
    .metric-card .desc {{ font-size: 0.85em; opacity: 0.8; margin-top: 5px; }}

    /* 文本 */
    .text-block {{ margin: 10px 0; color: #555; }}
    .highlight {{ background: #fff3cd; padding: 2px 6px; border-radius: 3px; font-weight: 600; }}

    /* 预测统计 */
    .pred-stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 15px 0; }}
    .pred-stat {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border-left: 4px solid #0f3460; }}
    .pred-stat .val {{ font-size: 1.4em; font-weight: 700; color: #1a1a2e; }}
    .pred-stat .lbl {{ font-size: 0.8em; color: #777; }}

    .footer {{ text-align: center; padding: 30px; color: #999; font-size: 0.85em; }}
</style>
</head>
<body>
<div class="container">

    <!-- 封面 -->
    <div class="cover">
        <h1>信用风险模型报告</h1>
        <div class="subtitle">Give Me Some Credit — Consumer Credit Risk Modeling</div>
        <div class="date">2026年7月</div>
    </div>

    <!-- 核心指标 -->
    <div class="section">
        <h2>核心指标</h2>
        <div class="metrics">
            <div class="metric-card">
                <div class="num">{summary_dict.get('最佳模型', 'XGBoost')}</div>
                <div class="desc">最佳模型</div>
            </div>
            <div class="metric-card">
                <div class="num">{summary_dict.get('验证集AUC', '0.8695')}</div>
                <div class="desc">验证集 AUC</div>
            </div>
            <div class="metric-card">
                <div class="num">{summary_dict.get('验证集KS', '0.5833')}</div>
                <div class="desc">验证集 KS</div>
            </div>
            <div class="metric-card">
                <div class="num">{summary_dict.get('交叉验证AUC', '0.8650')}</div>
                <div class="desc">交叉验证 AUC</div>
            </div>
        </div>
    </div>

    <!-- 1. 数据概览 -->
    <div class="section">
        <h2>1. 数据概览</h2>
        <table class="summary-table">
            {summary_rows}
        </table>
        <p class="text-block" style="margin-top:15px;">
            数据集来源于 Kaggle "Give Me Some Credit" 竞赛，目标是预测借款人未来两年内发生90天以上逾期违约的概率。
            训练集包含 <b>150,000</b> 条记录，测试集 <b>101,503</b> 条，整体违约率为 <b>6.68%</b>，属于典型的不平衡分类问题。
            原始特征10个，经特征工程衍生后共17个特征用于建模。
        </p>
    </div>

    <!-- 2. EDA -->
    <div class="section">
        <h2>2. 探索性数据分析</h2>
        <p class="text-block">
            目标变量严重失衡（93.32% 未违约 vs 6.68% 违约），建模时需采用 class_weight 或 scale_pos_weight 进行平衡处理。
            MonthlyIncome 缺失率约 19.8%，NumberOfDependents 缺失率约 2.6%，均用中位数填充。
        </p>
        {img_html('01_target_distribution.png', '目标变量分布')}
        {img_html('02_feature_distribution.png', '各特征分布直方图')}
        {img_html('03_correlation_heatmap.png', '特征相关性热力图')}
        {img_html('04_default_vs_nodefault.png', '违约 vs 非违约 特征分布对比')}
    </div>

    <!-- 3. 数据预处理与特征工程 -->
    <div class="section">
        <h2>3. 数据预处理与特征工程</h2>
        <h3>缺失值处理</h3>
        <table class="data-table">
            <tr><th>字段</th><th>缺失数量</th><th>缺失率</th><th>处理方式</th></tr>
            <tr><td>MonthlyIncome</td><td>29,731</td><td>19.82%</td><td>中位数填充</td></tr>
            <tr><td>NumberOfDependents</td><td>3,924</td><td>2.62%</td><td>中位数填充</td></tr>
        </table>
        <h3>异常值处理</h3>
        <table class="data-table">
            <tr><th>字段</th><th>异常情况</th><th>处理方式</th></tr>
            <tr><td>age</td><td>存在0岁、>100岁</td><td>截断至 [18, 100]</td></tr>
            <tr><td>逾期次数（3列）</td><td>存在96、98等异常值</td><td>截断至最大20</td></tr>
            <tr><td>RevolvingUtilizationOfUnsecuredLines</td><td>极端值达数万</td><td>截断至最大10</td></tr>
            <tr><td>DebtRatio</td><td>极端值达数千</td><td>截断至最大10</td></tr>
        </table>
        <h3>衍生特征（7个）</h3>
        <table class="data-table">
            <tr><th>特征名</th><th>说明</th></tr>
            <tr><td>TotalPastDue</td><td>三类逾期次数之和</td></tr>
            <tr><td>HasPastDue</td><td>是否有任何逾期记录（0/1）</td></tr>
            <tr><td>SevereLateCount</td><td>90天以上逾期次数</td></tr>
            <tr><td>TotalCreditLines</td><td>信贷总额度（开放信贷+不动产贷款）</td></tr>
            <tr><td>RealEstateRatio</td><td>不动产贷款占比</td></tr>
            <tr><td>IncomePerDependent</td><td>人均收入（月收入/(抚养人数+1)）</td></tr>
            <tr><td>LogMonthlyIncome</td><td>月收入对数变换</td></tr>
        </table>
    </div>

    <!-- 4. 模型对比 -->
    <div class="section">
        <h2>4. 模型训练与对比</h2>
        <p class="text-block">训练了4种模型，使用80/20分层划分，评估指标包括 AUC、KS、Brier Score 和 F1。</p>
        {model_table}
        <p class="text-block" style="margin-top:15px;">
            <b>XGBoost</b> 以 AUC=0.8695、KS=0.5833 的表现取得最优结果。LightGBM 紧随其后（AUC=0.8685）。
            两者均显著优于逻辑回归（AUC=0.8602），说明树模型在捕捉特征非线性关系方面具有优势。
        </p>
    </div>

    <!-- 5. 模型评估可视化 -->
    <div class="section">
        <h2>5. 模型评估可视化</h2>
        {img_html('05_roc_curves.png', 'ROC 曲线对比')}
        {img_html('06_pr_curves.png', 'Precision-Recall 曲线对比')}
        {img_html('07_ks_curve.png', 'KS 曲线（最佳模型）')}
        {img_html('08_feature_importance.png', '特征重要性 Top15')}
        {img_html('09_confusion_matrix.png', '混淆矩阵（最优阈值）')}
    </div>

    <!-- 6. 交叉验证 -->
    <div class="section">
        <h2>6. 交叉验证</h2>
        <p class="text-block">
            对最佳模型 XGBoost 进行3折交叉验证，AUC 均值为 <b>0.8650 ± 0.0022</b>，
            各折之间波动很小，说明模型泛化能力稳定，未出现过拟合。
        </p>
    </div>

    <!-- 7. 测试集预测 -->
    <div class="section">
        <h2>7. 测试集违约概率预测</h2>
        <p class="text-block">使用最佳模型 XGBoost 对测试集 101,503 条记录进行违约概率预测，结果保存至 <code>prediction_results.csv</code>。</p>
        <div class="pred-stats">
            <div class="pred-stat"><div class="val">{pred['Probability'].mean():.4f}</div><div class="lbl">平均概率</div></div>
            <div class="pred-stat"><div class="val">{pred['Probability'].median():.4f}</div><div class="lbl">中位数</div></div>
            <div class="pred-stat"><div class="val">{pred['Probability'].min():.4f}</div><div class="lbl">最小值</div></div>
            <div class="pred-stat"><div class="val">{pred['Probability'].max():.4f}</div><div class="lbl">最大值</div></div>
        </div>
        <p class="text-block">
            预测概率均值为 0.3207，高于训练集实际违约率 0.0668，这是因为模型使用了 scale_pos_weight 平衡正负样本，
            使得预测概率更倾向于"风险倾向"而非"绝对概率"。在实际应用中，可通过等 Platt Scaling 或 Isotonic Calibration 校准概率。
        </p>
        {img_html('10_prediction_distribution.png', '测试集预测违约概率分布')}
    </div>

    <!-- 8. 结论与建议 -->
    <div class="section">
        <h2>8. 结论与建议</h2>
        <h3>模型结论</h3>
        <ul style="margin: 10px 0 10px 25px; color: #555;">
            <li>XGBoost 模型 AUC=0.8695、KS=0.5833，具备良好的违约区分能力</li>
            <li>最重要的特征为 <b>NumberOfTimes90DaysLate</b>（90天以上逾期次数）和 <b>TotalPastDue</b>（总逾期次数），逾期历史是违约预测的核心信号</li>
            <li>RevolvingUtilizationOfUnsecuredLines（无担保信贷额度使用率）也是重要预测因子</li>
            <li>模型交叉验证 AUC=0.8650±0.0022，泛化稳定</li>
        </ul>
        <h3>业务建议</h3>
        <ul style="margin: 10px 0 10px 25px; color: #555;">
            <li>建议将客户按预测概率分为低/中/高/极高风险四档，实施差异化授信策略</li>
            <li>对有逾期记录（尤其90天以上）的客户加强贷后监控</li>
            <li>无担保信贷额度使用率 > 50% 的客户需重点关注</li>
            <li>模型可定期（如季度）使用新数据重新训练，保持预测时效性</li>
        </ul>
    </div>

    <div class="footer">
        Give Me Some Credit Risk Model Report | Generated by WorkBuddy | 2026-07
    </div>

</div>
</body>
</html>
"""

report_path = os.path.join(BASE_DIR, 'credit_risk_report.html')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"报告已生成: {report_path}")
