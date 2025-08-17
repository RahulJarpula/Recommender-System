# backend.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from predictor import predict
from evaluate import (
    precision_at_k,
    recall_at_k,
    intra_list_diversity,
    novelty,
    catalog_coverage,
    evaluate_model, train_test_split_ratings
)
from sklearn.preprocessing import normalize

# 1. Import all your train/predict functions
from models.knn_model       import train_knn,       predict_knn
from models.nmf_model       import train_nmf,       predict_nmf
from models.class_emb_model import train_class_emb, predict_class_emb
from models.reg_emb_model   import train_reg_emb,   predict_reg_emb
from models.ann_model       import train_ann,       predict_ann
## fine tuning ##
def get_feedback_scores(feedback_csv_path="data/user_feedback_log.csv"):
    try:
        df = pd.read_csv(feedback_csv_path)
        df["rating"] = df["feedback"].map({"👍": 1, "👎": 0})
        item_scores = df.groupby("item_id")["rating"].mean().to_dict()
        return item_scores
    except Exception as e:
        print(f"[Feedback Error] {e}")
        return {}
    
## ---------------##

def get_feedback_scores(feedback_csv_path="data/user_feedback_log.csv"):
    try:
        df = pd.read_csv(feedback_csv_path)
        df["rating"] = df["feedback"].map({"👍": 1, "👎": 0})
        item_scores = df.groupby("item_id")["rating"].mean().to_dict()
        return item_scores
    except Exception as e:
        print(f"[Feedback Error] {e}")
        return {}

def feedback_rerank(scores_dict, item_feedback_dict, alpha=0.8, beta=0.2):
    final_scores = {}
    for item, sim_score in scores_dict.items():
        feedback_score = item_feedback_dict.get(item, 0.5)
        final_scores[item] = alpha * sim_score + beta * feedback_score
    return final_scores
## ----------------- ##

# 2. Optional: a single place to load common CSVs
def load_data(embed_type="bow"):
    """
    Load your core data files once.
    You can then pass these DataFrames into train_* functions 
    instead of re-reading CSVs inside each one.
    """
    # ratings_df    = pd.read_csv()
    # user_emb_df   = pd.read_csv("data/user_emb.csv")
    # item_emb_df   = pd.read_csv("data/item_emb.csv")
    # return ratings_df, user_emb_df, item_emb_df
    ratings_df = pd.read_csv("data/ratings.csv")
    train_df, test_df, test_users = train_test_split_ratings(ratings_df)
    train_df.to_csv("data/train_ratings.csv", index=False)  # for masking

    user_emb_path = "data/user_emb.csv" if embed_type == "bow" else "data/user_bert_emb.csv"
    item_emb_path = "data/item_emb.csv" if embed_type == "bow" else "data/course_bert_emb.csv"

    user_emb_df = pd.read_csv(user_emb_path)
    item_emb_df = pd.read_csv(item_emb_path)

    return train_df, test_df, test_users, user_emb_df, item_emb_df

MODELS = {
    "KNN":       (train_knn,       predict_knn),
    "NMF":       (train_nmf,       predict_nmf),
    "ClassEmbd": (train_class_emb, predict_class_emb),
    "RegEmbd":   (train_reg_emb,   predict_reg_emb),
    "ANN":       (train_ann,       predict_ann),
}


def train(model_name: str, **params):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    
    train_fn, _ = MODELS[model_name]
    #params["use_feedback"] = use_feedback
    if model_name == "KNN":
        return train_fn(
            ratings_df=params["ratings_df"],
            n_neighbors=params.get("n_neighbors", 10),
            metric=params.get("metric", "cosine"),
            user_based=params.get("user_based", False)
        )

    elif model_name == "NMF":
        return train_fn(
            ratings_df=params["ratings_df"],
            #rating_path=params["rating_path"],
            #n_factors=params.get("n_factors", 50),
            #n_epochs=params.get("n_epochs", 15)
        )

    elif model_name == "ClassEmbd":
        return train_fn(
            rating_path=params["rating_path"],
            user_emb_path=params["user_emb_path"],
            item_emb_path=params["item_emb_path"],
            #model_type=params["model_type"],
            test_size=params["test_size"],
            random_state=params["random_state"]
        )

    elif model_name == "RegEmbd":
        return train_fn(
            rating_path=params["rating_path"],
            user_emb_path=params["user_emb_path"],
            item_emb_path=params["item_emb_path"],
            #test_size=params["test_size"],
            random_state=params["random_state"]
        )

    elif model_name == "ANN":
        return train_fn(
            rating_path=params["rating_path"],
            embed_dim=params["embed_dim"],
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            lr=params["lr"],
            device=params.get("device", "cpu")
        )

    else:
        return train_fn(**params)

def predict(model_name: str, **params):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    user_id = params.pop("user_id")
    top_n   = params.pop("top_n", 10)

    if model_name == "KNN":
        params.pop("rating_path", None)
        return predict_knn(user_id=user_id, top_n=top_n)

    elif model_name == "NMF":
        return predict_nmf(
            user_id=user_id,
            top_n=top_n,
            #ratings_df=params["rating_path"]
        )


    elif model_name == "ClassEmbd":
        return predict_class_emb(
            user_id=user_id,
            top_n=top_n,
            rating_path=params["rating_path"],
            user_emb_path=params["user_emb_path"],
            item_emb_path=params["item_emb_path"],
            #model_type=params["model_type"],
            #test_size=params["test_size"],
            #random_state=params["random_state"]
        )

    elif model_name == "RegEmbd":
        return predict_reg_emb(
            user_id=user_id,
            top_n=top_n,
            rating_path=params["rating_path"],
            user_emb_path=params["user_emb_path"],
            item_emb_path=params["item_emb_path"],
            #test_size=params["test_size"],
            #random_state=params["random_state"]
        )

    elif model_name == "ANN":
        return predict_ann(
            user_id=user_id,
            top_n=top_n,
            rating_path=params["rating_path"],
            device=params.get("device", "cpu")
        )


    else:
        # fallback for future models
        _, pred_fn = MODELS[model_name]
        # item_feedback_dict = get_feedback_scores()
        # reranked_scores = feedback_rerank(scores_dict, item_feedback_dict)
        # sorted_items = sorted(reranked_scores.items(), key=lambda x: x[1], reverse=True)
        return pred_fn(**params)
    # Example: scores_dict = {"BC0103EN": 0.81, "DA0101EN": 0.77, ...}



  