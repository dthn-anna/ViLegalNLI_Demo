import pandas as pd

df = pd.read_excel("Dataset/Train.xlsx")

# Ép toàn bộ column object về string
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype(str)
        
df.to_parquet("Dataset/Premise.parquet", index=False)