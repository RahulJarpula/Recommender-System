import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from models.knn_model       import train_knn,       predict_knn
from models.nmf_model       import train_nmf,       predict_nmf
from models.class_emb_model import train_class_emb, predict_class_emb
from models.reg_emb_model   import train_reg_emb,   predict_reg_emb
from models.ann_model       import train_ann,       predict_ann
MODELS = {
    "KNN":       (train_knn,       predict_knn),
    "NMF":       (train_nmf,       predict_nmf),
    "ClassEmbd": (train_class_emb, predict_class_emb),
    "RegEmbd":   (train_reg_emb,   predict_reg_emb),
    "ANN":       (train_ann,       predict_ann),
}

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
            ratings_df=params["rating_path"]
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
        return pred_fn(**params)



