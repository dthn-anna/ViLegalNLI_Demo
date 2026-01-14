import numpy as np

def cosine_retrieve(
    query_embedding,
    premise_embeddings,
    top_k=100
):
    scores = np.dot(premise_embeddings, query_embedding)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return top_idx, scores[top_idx]
