from pathlib import Path
 
import pandas as pd
 
DATA_PATH = Path("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
 
 
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)
 
 
if __name__ == "__main__":
    df = load_data()
    print("shape:", df.shape) 
    print("\n離職比例:")
    print(df["Attrition"].value_counts(normalize=True).round(3))  # 預期 No≈0.839 / Yes≈0.161
    print("\n缺失值總數:", int(df.isna().sum().sum()))
    nunique = df.nunique()
    print("\n常數欄位:", nunique[nunique == 1].index.tolist())
    print("可能的識別碼欄位:", [c for c in df.columns if "Number" in c or "ID" in c])