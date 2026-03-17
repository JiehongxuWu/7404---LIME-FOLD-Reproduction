import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.impute import SimpleImputer
from ucimlrepo import fetch_ucirepo
from .mdlp_discretizer import MDLPDiscretizer


heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
y = heart_disease.data.targets
# 二分类标签：是否患病（num > 0）
y_binary = (y['num'] > 0).astype(np.int64)


imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)


numeric_cols = ['trestbps', 'chol', 'oldpeak', 'ca']
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

# 数值特征离散化（对齐论文引用的 MDLPC/MDLP 思想：Fayyad & Irani, 1993）
X_numeric = X_imputed[numeric_cols].copy()
mdlp = MDLPDiscretizer(min_depth=1, max_depth=6, min_samples=10)
X_numeric_arr = X_numeric.to_numpy(dtype=np.float64)
X_numeric_discrete_arr = mdlp.fit_transform(X_numeric_arr, y_binary.to_numpy())
X_numeric_discrete = pd.DataFrame(
    X_numeric_discrete_arr.astype(np.int64),
    columns=numeric_cols,
    index=X_numeric.index,
)


X_onehot = pd.get_dummies(X_imputed[categorical_cols], columns=categorical_cols,
                          prefix=categorical_cols, drop_first=False)




def build_rename_map(columns, prefix, name_mapping):
    rename_dict = {}
    for col in columns:
        if col.startswith(prefix + '_'):
            suffix = col.split('_', 1)[1]
            # 注意：str.rstrip('.0') 会把末尾所有 '.' 或 '0' 都剥掉（例如 '0.0' 会变成 ''）
            # 这里我们只去掉“精确的后缀 .0”，以把 '0.0' -> '0', '1.0' -> '1'
            base_suffix = suffix[:-2] if suffix.endswith('.0') else suffix
            new_name = name_mapping.get(base_suffix, col)
            if new_name != col:
                rename_dict[col] = new_name
    return rename_dict


cp_map = {'1': 'chest_pain_1', '2': 'chest_pain_2', '3': 'chest_pain_3', '4': 'chest_pain_4'}
thal_map = {'3': 'thal_3', '6': 'thal_6', '7': 'thal_7'}
slope_map = {'1': 'slope_1', '2': 'slope_2', '3': 'slope_3'}
sex_map = {'0': 'sex_0', '1': 'sex_1'}
fbs_map = {'0': 'fbs_0', '1': 'fbs_1'}
restecg_map = {'0': 'restecg_0', '1': 'restecg_1', '2': 'restecg_2'}
exang_map = {'0': 'exercise_induced_angina_0', '1': 'exercise_induced_angina_1'}


rename_dict = {}
rename_dict.update(build_rename_map(X_onehot.columns, 'cp', cp_map))
rename_dict.update(build_rename_map(X_onehot.columns, 'thal', thal_map))
rename_dict.update(build_rename_map(X_onehot.columns, 'slope', slope_map))
rename_dict.update(build_rename_map(X_onehot.columns, 'sex', sex_map))
rename_dict.update(build_rename_map(X_onehot.columns, 'fbs', fbs_map))
rename_dict.update(build_rename_map(X_onehot.columns, 'restecg', restecg_map))
rename_dict.update(build_rename_map(X_onehot.columns, 'exang', exang_map))


X_onehot.rename(columns=rename_dict, inplace=True)


numeric_rename = {
    'ca': 'major_vessels',
    'chol': 'serum_cholestoral',
    'trestbps': 'blood_pressure',
    'oldpeak': 'oldpeak',
}

numeric_rename = {k: v for k, v in numeric_rename.items() if k in X_numeric_discrete.columns}
X_numeric_discrete.rename(columns=numeric_rename, inplace=True)


X_final = pd.concat([X_numeric_discrete, X_onehot], axis=1)



model = xgb.XGBClassifier(
    objective='binary:logistic',
    importance_type='weight',       
    random_state=42,
    n_estimators=600,
    max_depth=15,
    learning_rate=0.02,
    subsample=0.6,
    colsample_bytree=0.6,
    reg_lambda=3,
    reg_alpha=0.2,
    min_child_weight=2,
    eval_metric='logloss',
    use_label_encoder=False
)
model.fit(X_final, y_binary)

# ---------- 保存 Step1 产物（CSV + 模型 JSON）----------
# 约定输出位置：
# - data/processed/X_final.csv
# - data/processed/y_binary.csv
# - results/models/xgb_model.json
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "results" / "models"
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

X_final.to_csv(DATA_PROCESSED_DIR / "X_final.csv", index=False)
pd.Series(y_binary, name="y_binary").to_csv(DATA_PROCESSED_DIR / "y_binary.csv", index=False)
model.save_model(str(MODELS_DIR / "xgb_model.json"))


score_dict = model.get_booster().get_score(importance_type='weight')
model_feature_names = model.get_booster().feature_names



importance_dict = {}
for key, value in score_dict.items():
    if key.startswith('f') and key[1:].isdigit():
        idx = int(key[1:])
        feat_name = model_feature_names[idx]
    else:
        feat_name = key
    importance_dict[feat_name] = value


total = sum(importance_dict.values())
relative_importance = {k: v / total for k, v in importance_dict.items()}


sorted_items = sorted(relative_importance.items(), key=lambda x: x[1], reverse=True)
sorted_names = [item[0] for item in sorted_items]
sorted_values = [item[1] for item in sorted_items]


plt.figure(figsize=(12, 10))
bars = plt.barh(range(len(sorted_values)), sorted_values, align='center')
plt.yticks(range(len(sorted_values)), sorted_names)
plt.xlabel('Relative Importance (Percentage)')
plt.title('XGBoost Feature Relative Importance (Cleveland Dataset)')
plt.gca().invert_yaxis()


for i, (bar, val) in enumerate(zip(bars, sorted_values)):
    plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
             f'{val:.2%}', va='center', ha='left', fontsize=9)

plt.tight_layout()
# Save figure for reporting (Step1 output)
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
fig_path = FIGURES_DIR / "xgb_feature_importance.png"
plt.savefig(fig_path, dpi=150)
plt.show()
print(f"Step1 图已保存到: {fig_path}")