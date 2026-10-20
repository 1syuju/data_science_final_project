# 期末專題報告:員工離職預測

## 1. 簡介

### 1.1 研究動機
員工離職可能造成企業招募與培訓成本增加、組織知識流失，以及團隊工作效率下降。
若能透過機器學習提早辨識具有較高離職風險的員工，企業便能依據預測結果評估是否採取適當的留才措施,降低不必要的人力資源支出。
本專題將員工離職預測視為表格型二元分類問題，以 `Attrition` 作為預測目標，分為離職(Yes)與留任(No)。
由於資料集中離職員工約占 16.1%，存在類別不平衡問題，因此除了比較不同分類模型的預測能力，也需要探討不平衡處理方式對模型表現的影響。
此外，員工離職預測涉及實際的人力資源決策，因此模型不應只追求預測分數，也需要考慮模型可解釋性、不同人口群體間的預測差異，以及錯誤預測所產生的成本。

### 1.2 研究問題
1. **預測能力**:在中度不平衡的資料上，哪一類模型與哪一種不平衡處理方式(class weight、SMOTE、下採樣)的預測效果最好?
2. **可解釋性**:哪些特徵對離職預測最重要?模型的判斷是否符合 HR 直覺(例如加班頻率高、薪資滿意度低者離職率較高)?
3. **公平性**:模型對不同性別、年齡層的預測結果是否存在系統性差異?
4. **決策成本**:「誤判留任」與「誤判離職」的成本不對等時，最佳決策閾值為何?
5. **方法嚴謹性**:資料洩漏會讓驗證分數虛高多少?

### 1.3 背景知識
- **類別不平衡與評估指標**:離職約占 16.1%,若全部預測留任，Accuracy 可達約 84% 但完全抓不到離職者,所以以 ROC AUC 為主要指標，搭配 PR AUC 與離職類的 Recall / Precision / F1。PR AUC 對少數類更敏感。
- **不平衡處理方法**:class weight(調整損失權重)、SMOTE(合成少數類樣本)、下採樣(減少多數類樣本)，以及事後調整決策閾值。
- **資料洩漏(data leakage)**:任何需要從資料「學習」的轉換(編碼、標準化、填補、重抽樣、群組平均)若在切分前用全資料計算，驗證集資訊會洩漏進訓練，導致分數虛高。解法是把轉換包進 Pipeline，只在訓練折內擬合。
- **模型**:Logistic Regression(可解釋基準)、Decision Tree(易視覺化)、Random Forest(對小資料穩定)、XGBoost / LightGBM(梯度提升，預期表現最佳)。
- **可解釋性**:Permutation Importance 衡量整體重要性;SHAP 提供整體(Beeswarm)與單一員工(Waterfall)層級的特徵貢獻。
- **公平性指標**:比較各群體的預測離職率、False Positive Rate(FPR)與 False Negative Rate(FNR)，檢視是否有系統性偏差。
- **成本敏感決策**:FN(實際離職但預測留任)的成本高於 FP，依成本比例選擇期望成本最低的閾值，而不是固定用 0.5。

### 1.4 專案目標
1. 比較 Logistic Regression、Decision Tree、Random Forest、XGBoost、LightGBM 的預測表現，以 PR AUC 與 Recall 為重點。
2. 比較不處理、class weight、SMOTE、下採樣四種不平衡處理方式。
3. 建立不會資料洩漏的 Pipeline,並量化洩漏造成的驗證分數虛高幅度。
4. 用 Permutation Importance 與 SHAP 檢驗模型判斷是否符合 HR 直覺。
5. 分析不同性別與年齡群體的預測離職率、FPR、FNR。
6. 在假設成本比例下找出期望成本最低的決策閾值，並說明實務意涵。

---

## 2. 資料集

### 2.1 資料來源
本專題使用 Kaggle 上的 IBM HR Analytics Employee Attrition & Performance 資料集
https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

- 資料筆數:1,470 筆
- 原始特徵數:35 個欄位
- 離職比例:約 16.1%; 留任比例:約 83.9%(約 5:1)

**欄位說明**

| 欄位類型 | 欄位範例 |
|---|---|
| 人口背景 | 年齡(Age)、性別(Gender)、婚姻狀態(MaritalStatus)、教育程度(Education) |
| 薪資與福利 | 月薪(MonthlyIncome)、薪資漲幅(PercentSalaryHike)、股票選擇權等級(StockOptionLevel) |
| 工作條件 | 部門(Department)、職位(JobRole)、職級(JobLevel)、是否加班(OverTime)、出差頻率(BusinessTravel) |
| 滿意度指標 | 工作滿意度(JobSatisfaction)、環境滿意度(EnvironmentSatisfaction)、工作生活平衡(WorkLifeBalance)、工作投入感(JobInvolvement) |
| 職涯歷程 | 在職年數(YearsAtCompany)、上次升遷距今(YearsSinceLastPromotion)、共事過的公司數(NumCompaniesWorked) |
| 目標 | Attrition(Yes = 離職，No = 留任) |

此資料集包含員工人口背景、薪資福利、工作條件、滿意度及職涯歷程等資訊，適合用來研究員工離職預測與模型解釋。
但這是 IBM 提供的虛構員工資料，適合用於教學與方法驗證，不能直接視為真實企業員工的離職規律。

==滿意度指標感覺可以做成 weighted function ?==

### 2.2 已知挑戰
- 樣本數少(1,470 筆)且類別不平衡，離職者僅約 237 人，易過擬合，且每折驗證集的少數類樣本很少，分數變異大。
- 類別型與數值型特徵混合，含有序類別(Education、JobLevel 等)，需比較編碼方式。
- 含常數欄位與識別碼欄位，需在 EDA 階段移除。

### 2.3 資料處理
- 移除 EmployeeCount、Over18、StandardHours(常數)與 EmployeeNumber(員工編號無預測意義)。

| col |  |
|---|---|
| EmployeeCount	| 這一列代表幾位員工	全部都是 1 |
| Over18 | 是否年滿 18 歲	全部都是 Y |
| StandardHours |	標準工時	全部都是 80 |

- 依欄位性質處理數值與類別特徵:無序類別用 One-hot、有序類別保留數值順序，數值特徵不做 log 轉換；Logistic Regression 使用標準化，樹模型不做標準化。
- 建立四項衍生特徵:
  - 薪資壓力:月薪 / 同職級平均月薪
  - 職涯停滯:上次升遷距今 / 在職總年數(分母為 0 時設為 0)
  - 加班出差交互項(`workload`):加班 × 出差頻率，屬於交互特徵，不代表完整工作負荷
  - 滿意度綜合分數:工作滿意度、環境滿意度、工作生活平衡的等權平均
- 年齡分箱(`<30`、`30–39`、`40+`)僅供公平性分析使用，涵蓋資料中的 18 歲以上員工。
- 使用 Pipeline 整合前處理、特徵工程與模型。**「同職級平均月薪」與 SMOTE 等需要統計量的步驟都必須只在訓練折內計算**，否則會造成洩漏。

### 2.4 評估方式
- 以 80/20 分層切分出測試集，只在最後報告一次;其餘 80% 做 Stratified 10-fold Cross-Validation 進行模型選擇與調參。
  另外，使用重複分層 10 折(Repeated Stratified 10-fold)降低變異。
- 主要指標為 ROC AUC，搭配 PR AUC、離職類 Recall、Precision 與 F1
- Accuracy 僅作為輔助指標。
- 公平性分析不直接依賴只有 47 位離職者的測試集；模型選定後，使用 Stratified K-fold 的折外預測，再依性別與年齡群組比較預測離職率、FPR 與 FNR。
- Optuna 只對表現最好的一兩個模型調參(約 30–50 次試驗)，其餘模型使用預設值。

---

## 3. 目前進度

### 3.1 已完成
- [x] 看過資料集(欄位、規模、類別比例)
- [x] 確定主題與研究問題
- [x] 完成 EDA、前處理、特徵工程、評估方式與實驗設計的規劃

### 3.2 實驗設計(待執行)

| 實驗 | 變因 | 目的 |
|---|---|---|
| A | 不平衡處理(不處理 / class weight / SMOTE / 下採樣) | 比較對 PR AUC 與 Recall 的影響 |
| B | 模型(LR / DT / RF / XGBoost / LightGBM) | 找出最佳模型架構 |
| C | 正確流程 vs. 洩漏流程 | 量化驗證分數虛高幅度 |
| D | 決策閾值(0.3 / 0.4 / 0.5 / 最佳成本閾值) | 找出不同成本假設下的最佳決策 |

### 3.3 TO-DO

**資料與前處理**
- [x] 下載並載入資料集
- [x] EDA:欄位分布、缺失值、離職與留任差異、各類別離職率、相關性
- [x] 移除常數欄位與 EmployeeNumber
- [x] 確定編碼與數值轉換方式
- [x] 實作四項衍生特徵(組內統計量放進 Pipeline)
- [x] 切分測試集並固定隨機種子

**基準方法(naive methods)**
- [ ] 全部預測「留任」:確認 Accuracy 約 84%、Recall = 0
- [ ] Logistic Regression(標準化 + class weight):ROC AUC【待填】/ PR AUC【待填】
- [ ] Decision Tree(預設參數):【待填】
- [ ] Random Forest(預設參數):【待填】

**進階模型**
- [ ] XGBoost / LightGBM(預設參數):【待填】
- [ ] 對最佳一兩個模型用 Optuna 調參

**正式實驗**
- [ ] 實驗 A、B、C、D

**解釋與分析**
- [ ] Permutation Importance 與 SHAP
- [ ] 公平性分析(使用折外預測比較性別、年齡層的預測離職率、FPR、FNR)
- [ ] 成本敏感分析(FN 成本為 FP 的 3 倍、5 倍)

**交付**
- [ ] 整理可重現的 Jupyter Notebook
- [ ] 撰寫期末報告與圖表

---

## 4. 未來計畫


| 週次 | 日期 | 工作 | 產出 | 負責人 |
|---|---|---|---|---|
| 第 1 週 | 10/19–10/25 | 載入資料、EDA、固定前處理與特徵工程、切分測試集;跑完基準模型(LR、DT、RF) | EDA 圖表、可執行的 Pipeline、基準分數表 | 【待填】 |
| 第 2 週 | 10/26–11/1 | 完成實驗 A(不平衡處理);完成實驗 B(加入 XGBoost / LightGBM) | A、B 的比較表與圖 | 【待填】 |
| 第 3 週 | 11/2–11/8 | Optuna 調參;完成實驗 C(洩漏對照);Permutation Importance 與 SHAP | 調參結果、洩漏量化表、SHAP 圖 | 【待填】 |
| 第 4 週 | 11/9–11/15 | 公平性分析;完成實驗 D(成本敏感閾值);在測試集做最終評估 | 公平性表、PR 曲線、閾值與成本分析 | 【待填】 |
| 第 5 週 | 11/16–11/22 | 緩衝與收尾:整理 Notebook、撰寫報告與簡報 | 最終 Notebook、報告 | 【待填】 |

**風險與備案**
- 若 XGBoost / LightGBM 進度落後，先以 Random Forest 為主，梯度提升僅用預設參數。
- 若公平性分析時間不足，至少完成性別與年齡層的預測離職率與 FNR 比較。
- 離職樣本少導致分數變異大時，改用重複分層 10 折，並同時報告平均與標準差。
