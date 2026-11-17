"""
資料前處理(全部都在這個檔案)

==================== 怎麼用 ====================
在 notebook 裡只要這兩行:

    from src.preprocessing import get_train_test, make_preprocessor
    X_train, X_test, y_train, y_test = get_train_test()

建模型時把前處理和模型串成 Pipeline:

    from sklearn.pipeline import make_pipeline
    model = make_pipeline(make_preprocessor("linear"), LogisticRegression())  # 線性模型
    model = make_pipeline(make_preprocessor("tree"), RandomForestClassifier()) # 樹模型
    model.fit(X_train, y_train)

想看資料檢查結果:在專案根目錄執行  python -m src.preprocessing
================================================

整個流程:
  步驟 1  讀取原始 CSV
  步驟 2  移除沒用的欄位,目標轉成 0/1
  步驟 3  切 80% 訓練 / 20% 測試(切一次、存檔,全組共用)
  步驟 4  做衍生特徵、月薪取 log(FeatureEngineer)
  步驟 5  One-hot 類別欄位;線性模型再把數值標準化(make_preprocessor)

步驟 4、5 會「從資料學東西」(平均月薪、平均值/標準差、類別清單),
所以包在 Pipeline 裡,只用訓練資料 fit,避免資料洩漏。
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------- 設定
RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]  # 專案根目錄
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
SPLIT_PATH = ROOT / "data" / "split_indices.csv"
TARGET = "Attrition"

CONSTANT_COLS = ["EmployeeCount", "Over18", "StandardHours"]  # 每筆都一樣
ID_COLS = ["EmployeeNumber"]                                  # 員工編號
DROP_COLS = CONSTANT_COLS + ID_COLS

NOMINAL_COLS = ["Department", "EducationField", "JobRole", "MaritalStatus", "Gender"]
TRAVEL_ORDER = {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}
DERIVED_COLS = ["salary_ratio", "stagnation", "workload", "satisfaction_score"]

# 不放進模型、只留給公平性分析的欄位
# experiments.ipynb 第 2 節:拿掉 Gender 後 ROC AUC 只差 0.002,所以不讓模型使用性別
MODEL_DROP = ("Gender",)


# ---------------------------------------------------------------- 步驟 1:讀取
def load_raw(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


# ---------------------------------------------------------------- 步驟 2:清理
def split_xy(df: pd.DataFrame):
    """移除沒用的欄位;y = 1 代表離職。"""
    y = (df[TARGET] == "Yes").astype(int)
    X = df.drop(columns=DROP_COLS + [TARGET])
    return X, y


# ---------------------------------------------------------------- 步驟 3:切分
def load_saved_split(X, y, path=SPLIT_PATH):
    """照 split_indices.csv 的紀錄切出 train / test。"""
    s = pd.read_csv(path)
    tr = s.loc[s["split"] == "train", "row"]
    te = s.loc[s["split"] == "test", "row"]
    return X.loc[tr], X.loc[te], y.loc[tr], y.loc[te]


def get_train_test(path=SPLIT_PATH):
    """步驟 1~3 一次做完。split_indices.csv 不存在時才重新切並存檔。"""
    X, y = split_xy(load_raw())
    if Path(path).exists():
        return load_saved_split(X, y, path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    pd.concat([
        pd.DataFrame({"row": X_train.index, "split": "train"}),
        pd.DataFrame({"row": X_test.index, "split": "test"}),
    ]).sort_values("row").to_csv(path, index=False)
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------- 步驟 4:衍生特徵
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    OverTime → 0/1,BusinessTravel → 0/1/2,並新增四個特徵:
    - salary_ratio       月薪 / 同職級平均月薪(平均只用訓練資料算)
    - stagnation         上次升遷距今年數 / 在職年數(只算年資 ≥ 2 年者,其餘為 0;
                         年資太短的人「還沒到升遷時間」,不算停滯)
    - workload           加班 × 出差頻率
    - satisfaction_score 工作滿意度、環境滿意度、工作生活平衡的平均
    最後把 MonthlyIncome 換成 log(月薪),減少右偏(樹模型不受影響)。

    drop:不放進模型的欄位(例如 ["Gender"]),會在這一步移除。
    """

    def __init__(self, drop=()):
        self.drop = drop

    def fit(self, X, y=None):
        self.level_mean_ = X.groupby("JobLevel")["MonthlyIncome"].mean()
        self.global_mean_ = X["MonthlyIncome"].mean()
        return self

    def transform(self, X):
        X = X.copy()
        X["OverTime"] = (X["OverTime"] == "Yes").astype(int)
        X["BusinessTravel"] = X["BusinessTravel"].map(TRAVEL_ORDER)

        level_mean = X["JobLevel"].map(self.level_mean_).fillna(self.global_mean_)
        X["salary_ratio"] = X["MonthlyIncome"] / level_mean
        years = X["YearsAtCompany"].where(X["YearsAtCompany"] >= 2)
        X["stagnation"] = (X["YearsSinceLastPromotion"] / years).fillna(0.0)
        X["workload"] = X["OverTime"] * X["BusinessTravel"]
        X["satisfaction_score"] = X[
            ["JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance"]
        ].mean(axis=1)
        X["MonthlyIncome"] = np.log(X["MonthlyIncome"])
        return X.drop(columns=list(self.drop))


# ---------------------------------------------------------------- 步驟 5:編碼
def _nominal_present(X):
    """NOMINAL_COLS 中還留在資料裡的欄位(被 drop 的就跳過)。"""
    return [c for c in NOMINAL_COLS if c in X.columns]


def make_preprocessor(kind="linear", drop=MODEL_DROP) -> Pipeline:
    """
    kind="linear":Logistic Regression 用,類別 One-hot + 數值標準化
    kind="tree"  :決策樹 / RF / XGBoost / LightGBM 用,類別 One-hot,數值不動
    drop        :不放進模型的欄位,預設 MODEL_DROP
    """
    if kind not in ("linear", "tree"):
        raise ValueError("kind 只能是 'linear' 或 'tree'")

    numeric = StandardScaler() if kind == "linear" else "passthrough"
    encode = ColumnTransformer(
        [
            ("nominal", OneHotEncoder(handle_unknown="ignore", sparse_output=False), _nominal_present),
            ("num", numeric, make_column_selector(dtype_include="number")),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    return Pipeline([("features", FeatureEngineer(drop=drop)), ("encode", encode)])


# ---------------------------------------------------------------- 公平性分析用(不進模型)
def age_group(age: pd.Series) -> pd.Series:
    return pd.cut(age, bins=[0, 29, 39, 200], labels=["<30", "30-39", "40+"])


# ---------------------------------------------------------------- 直接執行:檢查一遍
if __name__ == "__main__":
    df = load_raw()
    print("原始資料:", df.shape, " 缺失值:", int(df.isna().sum().sum()))
    print("離職比例:", round((df[TARGET] == "Yes").mean(), 3))

    X_train, X_test, y_train, y_test = get_train_test()
    print("train:", X_train.shape, " 離職", int(y_train.sum()), "人")
    print("test :", X_test.shape, " 離職", int(y_test.sum()), "人")

    for kind in ("linear", "tree"):
        Xt = make_preprocessor(kind).fit_transform(X_train, y_train)
        print(f"{kind} 前處理後:", Xt.shape)
