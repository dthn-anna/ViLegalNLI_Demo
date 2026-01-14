import streamlit as st
import pandas as pd
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

from vector_search import cosine_retrieve

# ======================
# CONFIG
# ======================
NLI_MODEL_PATH = "Model"
EMBED_MODEL = "AITeamVN/Vietnamese_Embedding"

EMB_PATH = "Dataset/premise_embeddings.npy"
META_PATH = "Dataset/premise_meta.parquet"

THRESHOLD = 0.75
TOP_K_RETRIEVE = 30
TOP_K_SHOW = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="Legal Entailment Demo", layout="wide")

# ======================
# LOAD MODELS (CACHE)
# ======================
@st.cache_resource
def load_models():
    embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_PATH)
    nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_PATH)
    nli_model.to(DEVICE)
    nli_model.eval()

    return embedder, tokenizer, nli_model


@st.cache_data
def load_data():
    emb = np.load(EMB_PATH)
    meta = pd.read_parquet(META_PATH)
    return emb, meta


embedder, tokenizer, nli_model = load_models()
premise_embeddings, meta_df = load_data()

# ======================
# UI
# ======================
st.title("⚖️ HỆ THỐNG TRUY XUẤT VÀ SUY LUẬN PHÁP LÝ VIỆT NAM")

hypothesis = st.text_area(
    "Nhập giả thuyết pháp lý:",
    height=120
)

run_btn = st.button("🔍 Kiểm tra căn cứ pháp lý")

# ======================
# NLI
# ======================
def batch_nli(premises, hypothesis):
    hypotheses = [hypothesis] * len(premises)

    inputs = tokenizer(
        premises,
        hypotheses,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    ).to(DEVICE)

    with torch.no_grad():
        logits = nli_model(**inputs).logits

    probs = torch.softmax(logits, dim=1)

    # Bài toán 2 nhãn: ENTAILMENT vs NON-ENTAILMENT
    # Giả định model: [NON-ENTAILMENT, ENTAILMENT]
    return probs[:, 1].cpu().numpy()


def render_legal_citation(row):
    parts = []

    if pd.notna(row.get("Tag Point")):
        parts.append(f"Điểm {row['Tag Point']}")
    if pd.notna(row.get("Clause")):
        parts.append(f"Khoản {row['Clause']}")
    if pd.notna(row.get("Article")):
        parts.append(f"Điều {row['Article']}")

    citation = ", ".join(parts)

    law_info = []
    if pd.notna(row.get("Law Name")):
        law_info.append(row["Law Name"])
    if pd.notna(row.get("Law ID")):
        law_info.append(f"Số: {row['Law ID']}")
    if pd.notna(row.get("Law Date")):
        law_info.append(f"({row['Law Date']})")

    return citation, " ".join(law_info)


# ======================
# RUN
# ======================
if run_btn and hypothesis.strip():
    with st.spinner("⏳ Đang suy luận pháp lý..."):
        # 1️⃣ Embed hypothesis
        query_emb = embedder.encode(
            hypothesis,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # 2️⃣ Retrieve top-K premise
        idxs, sim_scores = cosine_retrieve(
            query_emb,
            premise_embeddings,
            top_k=TOP_K_RETRIEVE
        )

        candidates = meta_df.iloc[idxs].copy()
        candidates["similarity"] = sim_scores

        # 3️⃣ NLI trên từng premise
        entail_probs = batch_nli(
            candidates["Premise"].tolist(),
            hypothesis
        )
        candidates["entail_prob"] = entail_probs

        candidates = candidates.sort_values(
            "entail_prob", ascending=False
        )

        # 4️⃣ AGGREGATION → SUY LUẬN CUỐI
        # dùng top-3 premise mạnh nhất
        final_score = np.max(candidates.head(3)["entail_prob"])

        final_label = (
            "ENTAILMENT" if final_score >= THRESHOLD
            else "NON-ENTAILMENT"
        )

    st.divider()

    # ======================
    # RESULT
    # ======================
    if final_label == "ENTAILMENT":
        st.success("✅ **CÂU PHÁT BIỂU TRÊN LÀ ĐÚNG**")
        st.caption(f"Điểm suy luận cuối: **{final_score:.3f}**")

        st.subheader("📌 Nhập căn cứ pháp lý hỗ trợ")
        for _, r in candidates.head(TOP_K_SHOW).iterrows():
            citation, law_info = render_legal_citation(r)

            st.markdown(f"""
**Entailment:** `{r.entail_prob:.3f}`  
**Similarity:** `{r.similarity:.3f}`  

📌 **Căn cứ pháp lý:**  
{citation}  

📘 **Văn bản:**  
{law_info}  

> {r.Premise}
""")
            st.markdown("---")

    else:
        st.error("❌ **CÂU PHÁT BIỂU TRÊN KHÔNG ĐÚNG**")
        st.caption(f"Điểm suy luận cao nhất: **{final_score:.3f}**")

        st.subheader("🔍 Các điều luật liên quan (chưa đủ căn cứ)")
        for _, r in candidates.head(TOP_K_SHOW).iterrows():
            st.markdown(f"""
**Entailment:** `{r.entail_prob:.3f}`  
**Similarity:** `{r.similarity:.3f}`  

> {r.Premise}
""")
            st.markdown("---")
