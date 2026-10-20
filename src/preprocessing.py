"""
IBM HR Attrition 前處理:載入、移除欄位、衍生特徵、Pipeline、切分。

設計重點(對應提案的「避免資料洩漏」):
- 所有需要從資料「學習」的東西(同職級平均月薪、標準化的平均與標準差、
  One-hot 的類別集合)都只在 Pipeline.fit() 時、用訓練資料計算。
- 切分只做一次,索引存檔,所有人共用同一份。

"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.2
DATA_PATH = Path("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
SPLIT_PATH = Path("data/split_indices.csv")
TARGET = "Attrition"

# 常數欄位 + 員工編號(無預測意義)
CONSTANT_COLS = ["EmployeeCount", "Over18", "StandardHours"]
ID_COLS = ["EmployeeNumber"]
DROP_COLS = CONSTANT_COLS + ID_COLS

# BusinessTravel 有順序,手動編碼
TRAVEL_ORDER = {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}

# 無序類別 -> One-hot
NOMINAL_COLS = ["Department", "EducationField", "JobRole", "MaritalStatus", "Gender"]

# 公平性分析用的年齡分組(資料中 Age 範圍約 18–60,故最低組為 <30)
AGE_BINS = [0, 29, 39, 200]
AGE_LABELS = ["<30", "30-39", "40+"]


# ---------------------------------------------------------------- 載入與切分
def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """移除常數欄位與 EmployeeNumber,並把目標轉成 0/1(1 = 離職)。"""
    y = (df[TARGET] == "Yes").astype(int)
    X = df.drop(columns=DROP_COLS + [TARGET])
    return X, y


def age_group(age: pd.Series) -> pd.Series:
    """公平性分析用的年齡分組,不放進模型。"""
    return pd.cut(age, bins=AGE_BINS, labels=AGE_LABELS)


def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    save_path: Path | None = SPLIT_PATH,
):
    """80/20 分層切分,固定種子;同時把索引存成 CSV,方便全組共用。"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(
            [
                pd.DataFrame({"row": X_train.index, "split": "train"}),
                pd.DataFrame({"row": X_test.index, "split": "test"}),
            ]
        ).sort_values("row").to_csv(save_path, index=False)
    return X_train, X_test, y_train, y_test


def load_saved_split(X: pd.DataFrame, y: pd.Series, path: Path = SPLIT_PATH):
    """依存下來的索引重建切分(組員用這個,不要自己重切)。"""
    s = pd.read_csv(path)
    tr = s.loc[s["split"] == "train", "row"].to_numpy()
    te = s.loc[s["split"] == "test", "row"].to_numpy()
    return X.loc[tr], X.loc[te], y.loc[tr], y.loc[te]


# ---------------------------------------------------------------- 衍生特徵
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    
    加入四個衍生特徵,並把 OverTime / BusinessTravel 轉成數值。

    - salary_ratio:月薪 / 同職級(JobLevel)平均月薪。
      *職級平均只在 fit() 用訓練資料計算*,transform() 只套用,所以放進
      Pipeline 後交叉驗證每一折都是各自重算,不會洩漏。
    - stagnation:上次升遷距今年數 / 在職年數(在職年數為 0 時設為 0)。
    - workload:OverTime(0/1) × BusinessTravel(0/1/2)。
    - satisfaction_score:工作滿意度、環境滿意度、工作生活平衡的平均
      (待調整)。

    """

    SAT_COLS = ["JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance"]

    def fit(self, X: pd.DataFrame, y=None):
        self.level_mean_ = X.groupby("JobLevel")["MonthlyIncome"].mean()
        self.global_mean_ = float(X["MonthlyIncome"].mean())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["OverTime"] = (X["OverTime"] == "Yes").astype(int)
        X["BusinessTravel"] = X["BusinessTravel"].map(TRAVEL_ORDER)

        # 訓練中沒出現過的職級,退回訓練集整體平均
        level_mean = X["JobLevel"].map(self.level_mean_).fillna(self.global_mean_)
        X["salary_ratio"] = X["MonthlyIncome"] / level_mean

        years = X["YearsAtCompany"].replace(0, np.nan)
        X["stagnation"] = (X["YearsSinceLastPromotion"] / years).fillna(0.0)

        X["workload"] = X["OverTime"] * X["BusinessTravel"]
        X["satisfaction_score"] = X[self.SAT_COLS].mean(axis=1)
        return X

    def get_feature_names_out(self, input_features=None):  # pragma: no cover
        return None


DERIVED_COLS = ["salary_ratio", "stagnation", "workload", "satisfaction_score"]


# ---------------------------------------------------------------- Pipeline
def make_preprocessor(kind: str = "linear") -> Pipeline:
    """
    
    回傳「衍生特徵 + 編碼/轉換」的 Pipeline(尚未 fit)。

    kind="linear":給 Logistic Regression。數值標準化,類別 One-hot。
    kind="tree":給 DT / RF / XGBoost / LightGBM。不做轉換與標準化。
    其餘有序欄位(Education、JobLevel、各項滿意度、StockOptionLevel 等)資料本身
    就是數字 1–4 或 0–3,直接當有序數值保留。
    
    """
    if kind not in {"linear", "tree"}:
        raise ValueError("kind must be 'linear' or 'tree'")

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    if kind == "linear":
        numeric_cols = _numeric_cols_after_engineering()
        ct = ColumnTransformer(
            [
                ("nominal", ohe, NOMINAL_COLS),
                ("num", StandardScaler(), numeric_cols),
            ],
            verbose_feature_names_out=False,
        )
    else:
        numeric_cols = _numeric_cols_after_engineering()
        ct = ColumnTransformer(
            [
                ("nominal", ohe, NOMINAL_COLS),
                ("num", "passthrough", numeric_cols),
            ],
            verbose_feature_names_out=False,
        )
    ct.set_output(transform="pandas")
    return Pipeline([("features", FeatureEngineer()), ("encode", ct)])


# 原始數值欄位(移除常數與編號之後)+ 轉成數值的 OverTime/BusinessTravel + 衍生特徵
_RAW_NUMERIC = [
    "Age", "DailyRate", "DistanceFromHome", "Education", "EnvironmentSatisfaction",
    "HourlyRate", "JobInvolvement", "JobLevel", "JobSatisfaction", "MonthlyIncome",
    "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
]


def _numeric_cols_after_engineering() -> list[str]:
    return _RAW_NUMERIC + ["OverTime", "BusinessTravel"] + DERIVED_COLS
