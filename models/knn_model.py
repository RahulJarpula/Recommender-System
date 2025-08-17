import pickle
import pandas as pd
from surprise import KNNBasic
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

def train_knn(ratings_df: pd.DataFrame,
              n_neighbors: int = 10,
              metric: str = 'cosine',
              user_based: bool = False) -> None:
    """
    Fit a NearestNeighbors model on the user–item matrix and serialize it.
    """
    # 1. Build a Surprise dataset from the DataFrame
    reader = Reader(rating_scale=(ratings_df['rating'].min(),
                                  ratings_df['rating'].max()))
    data = Dataset.load_from_df(
        ratings_df[['user', 'item', 'rating']],
        reader
    )

    # 2. Split off a trainset (we ignore the test set here, but you can evaluate if you like)
    trainset, _ = train_test_split(data, test_size=0.3)

    # 3. Configure similarity options
    sim_options = {
        'name':      metric,
        'user_based': user_based
    }

    # 4. Initialize & fit the algorithm
    algo = KNNBasic(
        k=n_neighbors,
        min_k=1,
        sim_options=sim_options,
        verbose=True
    )
    algo.fit(trainset)
    
    # save both model and matrix for inference
    with open("models/knn_model.pkl", "wb") as f:
        pickle.dump(algo,f)


def predict_knn(user_id: int, top_n: int = 10, **kwargs) -> pd.Series:
    """
    Load trained KNN and return top_n recommendations for user_id.
    """
    with open("models/knn_model.pkl", "rb") as f:
        algo = pickle.load(f)
    trainset = algo.trainset

    # 2. Map raw user ID → inner UID (0…n_users-1)
    try:
        inner_uid = trainset.to_inner_uid(user_id)
    except ValueError:
        raise KeyError(f"User {user_id} not found in training data.")

    # 3. Build the list of candidate inner IIDs (all items minus already-rated)
    all_inner_iids   = list(trainset.all_items())                       # [0…n_items-1]
    rated_inner_iids = {iid for (iid, _) in trainset.ur[inner_uid]}     # items the user rated
    candidates       = [iid for iid in all_inner_iids if iid not in rated_inner_iids]

    # 4. Predict a rating for each candidate raw IID
    preds = []
    for iid in candidates:
        raw_iid = trainset.to_raw_iid(iid)             # convert back to your original item ID
        est     = algo.predict(user_id, raw_iid).est   # Surprise’s .predict returns a namedtuple
        preds.append((raw_iid, est))

    # 5. Sort by estimated rating descending, take top_n
    preds.sort(key=lambda x: x[1], reverse=True)
    top_items = [raw_iid for raw_iid, _ in preds[:top_n]]

    # 6. Return as a pandas Series for easy downstream merging/display
    return pd.Series(top_items, name="item")
