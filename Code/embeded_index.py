import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# File dữ liệu gốc ở dạng parquet
DATA_PATH = "Dataset/Premise.parquet"

# Đường dẫn để lưu file embedding
EMB_PATH = "Dataset/premise_embeddings.npy"

# Đường dẫn để lưu file metadata (không bao gồm embedding)
META_PATH = "Dataset/premise_meta.parquet"


# Mô hình embedding
EMBED_MODEL = "AITeamVN/Vietnamese_Embedding"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("🔹 Loading dataset...")
df = pd.read_parquet(DATA_PATH)

# Đảm bảo Premise là string
df["Premise"] = df["Premise"].astype(str)

print("🔹 Loading embedder...")
model = SentenceTransformer(EMBED_MODEL, device=device)

print("🔹 Encoding premises...")
embeddings = model.encode(
    df["Premise"].tolist(),
    batch_size=128,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("🔹 Saving embeddings...")
np.save(EMB_PATH, embeddings)

print("🔹 Saving metadata (KEEP Premise)...")
meta_cols = [
    "Law ID",
    "Law Name",
    "Law Date",
    "Article",
    "Clause",
    "Tag Point",
    "Premise"  
]

df[meta_cols].to_parquet(META_PATH, index=False)

print("✅ DONE")
