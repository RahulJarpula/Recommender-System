import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_class_emb(rating_path: str,
                    user_emb_path: str,
                    item_emb_path: str,
                    model_type: str = 'logistic',
                    test_size: float = 0.2,
                    random_state: int = 42,
                    **model_kwargs) -> None:
    """
    Train a classifier on user+item embeddings to predict rating classes.
    """
    # load data
    ratings = pd.read_csv(rating_path)
    uemb    = pd.read_csv(user_emb_path)
    iemb    = pd.read_csv(item_emb_path)
    
    df = (ratings
          .merge(uemb, on='user', how='left')
          .merge(iemb, on='item', how='left')
          .fillna(0))
    
    X = df.drop(columns=['user','item','rating'])
    y = LabelEncoder().fit_transform(df['rating'])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    
    if model_type == 'logistic':
        clf = LogisticRegression(**model_kwargs)
    else:
        clf = RandomForestClassifier(**model_kwargs)
    
    clf.fit(X_train, y_train)
    
    # you can optionally evaluate here (accuracy, f1, etc.)
    
    with open("models/class_emb_model.pkl", "wb") as f:
        pickle.dump((clf, X.columns), f)


def predict_class_emb(user_id: int,
                      rating_path: str,
                      user_emb_path: str,
                      item_emb_path: str,
                      top_n: int = 10) -> pd.Series:
    """
    Load classifier and return top_n items with highest predicted class-probability.
    """
    with open("models/class_emb_model.pkl", "rb") as f:
        clf, feature_cols = pickle.load(f)
    
    # reload embeddings & ratings
    ratings = pd.read_csv(rating_path)
    uemb    = pd.read_csv(user_emb_path)
    iemb    = pd.read_csv(item_emb_path)
    
    # build one-user basket of all candidate items
    user_vec = (pd.DataFrame({'user':[user_id] * len(iemb), 'item': iemb['item']})
                .merge(uemb, on='user')
                .merge(iemb, on='item')
                .fillna(0))
    
    X_pred = user_vec[feature_cols]
    probs  = clf.predict_proba(X_pred)[:, 1]  # assume class “positive” at index 1
    
    user_vec['prob'] = probs
    recs = (user_vec.sort_values('prob', ascending=False)
                   .head(top_n)['item'])
    return recs.reset_index(drop=True)
