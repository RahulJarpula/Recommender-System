import pickle
import pandas as pd
from surprise import NMF
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

def train_nmf(ratings_df: pd.DataFrame,
              n_components: int = 10,
              random_state: int = 42) -> None:
    """
    Fit an NMF model on the user–item matrix and serialize it.
    """
    reader = Reader(rating_scale=(ratings_df['rating'].min(),
                                  ratings_df['rating'].max()))
    data = Dataset.load_from_df(
        ratings_df[['user', 'item', 'rating']],
        reader
    )
    trainset, _ = train_test_split(data, test_size=0.3)
    algo = NMF(n_factors=20, n_epochs=50, reg_pu=0.06, reg_qi=0.06)
    algo.fit(trainset)
    
    with open("models/nmf_model.pkl", "wb") as f:
        pickle.dump(algo, f)

def predict_nmf(user_id: str, top_n: int = 10) -> pd.Series:
    """
    Load trained Surprise NMF and return top_n recommendations for user_id.
    """
    # 1. Load the serialized NMF algorithm
    with open("models/nmf_model.pkl", "rb") as f:
        algo = pickle.load(f)
    trainset = algo.trainset

    # 2. Map raw user ID → inner UID
    try:
        inner_uid = trainset.to_inner_uid(user_id)
    except ValueError:
        raise KeyError(f"User {user_id} not found in training data.")

    # 3. Build candidate list of all inner IIDs minus those already rated
    all_inner_iids   = list(trainset.all_items())
    rated_inner_iids = {iid for (iid, _) in trainset.ur[inner_uid]}
    candidates       = [iid for iid in all_inner_iids if iid not in rated_inner_iids]

    # 4. Predict a rating for each candidate
    preds = []
    for iid in candidates:
        raw_iid = trainset.to_raw_iid(iid)
        est     = algo.predict(user_id, raw_iid).est
        preds.append((raw_iid, est))

    # 5. Sort by estimated rating descending, then take top_n
    preds.sort(key=lambda x: x[1], reverse=True)
    top_items = [raw_iid for raw_iid, _ in preds[:top_n]]

    # 6. Return as a pandas Series
    return pd.Series(top_items, name="item")
