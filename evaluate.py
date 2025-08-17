import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
import itertools
from predictor import predict
from collections import defaultdict

def train_test_split_ratings(ratings_df):
    ratings_df = pd.read_csv("data/ratings.csv")

    # Split users into train/test
    users = ratings_df['user'].unique()
    np.random.seed(42)
    np.random.shuffle(users)

    split_idx = int(0.8 * len(users))
    train_users = users[:split_idx]
    test_users  = users[split_idx:]

    train_df = ratings_df[ratings_df['user'].isin(train_users)].copy()
    test_df  = ratings_df[ratings_df['user'].isin(test_users)].copy()

    # Save to disk (only if not already saved)
    if not os.path.exists("data/train_ratings.csv"):
        train_df.to_csv("data/train_ratings.csv", index=False)
    if not os.path.exists("data/test_ratings.csv"):
        test_df.to_csv("data/test_ratings.csv", index=False)

    return train_df, test_df, test_users
# -------------------
# Precision@k / Recall@k
# -------------------
def precision_at_k(preds, actuals, k=10):
    return np.mean([
        len(set(p[:k]) & set(a)) / k
        for p, a in zip(preds, actuals)
    ])

def recall_at_k(preds, actuals, k=10):
    return np.mean([
        len(set(p[:k]) & set(a)) / len(a) if a else 0
        for p, a in zip(preds, actuals)
    ])

# -------------------
# Intra-List Diversity
# -------------------
def intra_list_diversity(preds, item_embeddings_df):
    diversities = []
    for items in preds:
        if len(items) <= 1:
            continue
        embs = item_embeddings_df.loc[items].values
        sim = cosine_similarity(embs)
        diversity = 1 - np.mean(sim[np.triu_indices(len(embs), k=1)])
        diversities.append(diversity)
    return np.mean(diversities) if diversities else 0.0

# -------------------
# Inter-List Diversity
# -------------------
def inter_list_diversity(preds):
    all_pairs = list(itertools.combinations(preds, 2))
    divs = []
    for p1, p2 in all_pairs:
        if not p1 or not p2:
            continue
        sim = len(set(p1) & set(p2)) / len(set(p1) | set(p2))
        divs.append(1 - sim)
    return np.mean(divs) if divs else 0.0  # ✅ Fix here


# -------------------
# Novelty
# -------------------
def novelty(preds, item_popularity, k=10):
    all_logs = []
    for rec_list in preds:
        for item in rec_list[:k]:
            freq = item_popularity.get(item, 1e-6)
            all_logs.append(-np.log2(freq + 1e-9))  # Less frequent = higher novelty
    return abs(np.mean(all_logs)) if all_logs else 0.0

# -------------------
# Catalog Coverage
# -------------------
def catalog_coverage(preds, total_items):
    recommended = set(i for user in preds for i in user)
    return len(recommended) / total_items

def evaluate_model(model_name, user_ids, top_n, train_df, test_df, item_embeddings_df, item_popularity, embed_type="bow"):
    all_preds   = []
    all_actuals = []
    #print("👀 ratings_df columns:", ratings_df.columns.tolist())
    for user_id in user_ids:
        try:
            recs = predict(
                model_name=model_name,
                user_id=user_id,
                top_n=top_n,
                rating_path="data/train_ratings.csv",  # mask only seen in train
                user_emb_path="data/user_emb.csv" if embed_type == "bow" else "data/user_bert_emb.csv",
                item_emb_path="data/item_emb.csv" if embed_type == "bow" else "data/course_bert_emb.csv",
                device="cpu"
            )
        except Exception as e:
            print(f"[User {user_id}] Error during prediction: {e}")
            continue

        # Normalize format
        if isinstance(recs, pd.DataFrame):
            recs = recs["item"].tolist()
        elif isinstance(recs, np.ndarray):
            recs = recs.tolist()
        elif isinstance(recs, pd.Series):  # ✅ handle this too
            recs = recs.tolist()
        elif isinstance(recs, list):
            pass  # good
        else:
            print(f"[User {user_id}] Unknown prediction format: {type(recs)}")
            continue  # 🔁 skip this user

        # Now safe to define actual_items
        # print("🧪 DEBUG: test_df columns:", test_df.columns.tolist())
        # print("🧪 DEBUG: test_df sample rows:\n", test_df.head())

        actual_items = test_df[(test_df["user"] == user_id) & (test_df["rating"] >= 3)]["item"].tolist()

        print("User:", user_id)
        print("Actual items:", actual_items)
        print("Predicted items:", recs)
        print("Overlap:", set(actual_items) & set(recs))


        all_preds.append(recs)
        all_actuals.append(actual_items)

    # --- Compute Metrics ---
    metrics = {
        "Precision@10":      precision_at_k(all_preds, all_actuals, k=10),
        "Recall@10":         recall_at_k(all_preds, all_actuals, k=10),
        "Intra-list Div":    intra_list_diversity(all_preds, item_embeddings_df),
        "Inter-list Div":    inter_list_diversity(all_preds),
        "Novelty":           novelty(all_preds, item_popularity, k=10),
        "Catalog Coverage":  catalog_coverage(all_preds, total_items=len(item_popularity)),
    }

    return metrics


