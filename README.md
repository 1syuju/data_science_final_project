# 員工離職預測(資料科學期末專題,第 17 組)

用 IBM HR Analytics 資料集(1,470 筆、離職 16.1%)預測員工是否離職,
並檢驗不平衡處理、資料洩漏、決策閾值、可解釋性與公平性。

**完整報告:[ProgressReport.md](ProgressReport.md)**

## 主要結論

- 最終模型是 **Logistic Regression**:交叉驗證 ROC AUC 0.845、PR AUC 0.681;測試集 ROC AUC 0.813(95% CI 0.74–0.88)。
- 不平衡處理(class weight / SMOTE / 下採樣)都沒有提升 AUC,用**調整閾值**就能達到相同效果;FN 成本為 5 倍時閾值選 0.22。
- **SMOTE 放在切分前**會讓 PR AUC 虛高 0.26~0.47。
- 加班是最重要的因素;模型對 **40 歲以上員工**漏判較多(FNR 0.46)。

## 檔案結構

```
data/
  WA_Fn-UseC_-HR-Employee-Attrition.csv   原始資料
  split_indices.csv                       全組共用的 train/test 切分(不要自己重切)
src/
  preprocessing.py                        前處理(載入、切分、衍生特徵、編碼)
  eda.ipynb                               探索性分析
  experiments.ipynb                       所有實驗:模型、不平衡、洩漏、閾值、SHAP、公平性、測試集
reports/
  *.csv                                   各實驗結果表
  figures/*.png                           報告用的圖
  Group17_Exercise2.pdf                   期中進度報告
ProgressReport.md                         期末報告
```

## 怎麼執行

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS / Linux:source .venv/bin/activate
pip install -r requirements.txt
python -m src.preprocessing     # 檢查資料與切分
```

接著在 VS Code / Jupyter 選 `.venv` 的 kernel,依序執行 `src/eda.ipynb`、`src/experiments.ipynb`
(實驗約 3 分鐘)。結果會覆寫 `reports/` 裡的表與圖。

在自己的 notebook 使用前處理:

```python
from src.preprocessing import get_train_test, make_preprocessor
X_train, X_test, y_train, y_test = get_train_test()
```
