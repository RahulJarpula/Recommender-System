import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def train_reg_emb(rating_path: str,
                  user_emb_path: str,
                  item_emb_path: str,
                  test_size: float = 0.2,
                  random_state: int = 42) -> None:
    """
    Train a regression model on user+item embeddings to predict ratings.
    """
    ratings = pd.read_csv(rating_path)
    uemb    = pd.read_csv(user_emb_path)
    iemb    = pd.read_csv(item_emb_path)
    
    df = (ratings
          .merge(uemb, on='user', how='left')
          .merge(iemb, on='item', how='left')
          .fillna(0))
    
    X = df.drop(columns=['user','item','rating'])
    y = df['rating'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    
    # optionally evaluate (MSE, R²)
    
    with open("models/reg_emb_model.pkl", "wb") as f:
        pickle.dump((reg, X.columns), f)


def predict_reg_emb(user_id: int,
                    rating_path: str,
                    user_emb_path: str,
                    item_emb_path: str,
                    top_n: int = 10) -> pd.Series:
    """
    Load regression model and return top_n items with highest predicted rating.
    """
    with open("models/reg_emb_model.pkl", "rb") as f:
        reg, feature_cols = pickle.load(f)
    
    ratings = pd.read_csv(rating_path)
    uemb    = pd.read_csv(user_emb_path)
    iemb    = pd.read_csv(item_emb_path)
    
    user_vec = (pd.DataFrame({'user':[user_id]*len(iemb), 'item': iemb['item']})
                .merge(uemb, on='user')
                .merge(iemb, on='item')
                .fillna(0))
    
    X_pred = user_vec[feature_cols]
    preds  = reg.predict(X_pred)
    
    user_vec['pred_rating'] = preds
    recs = (user_vec.sort_values('pred_rating', ascending=False)
                   .head(top_n)['item'])
    return recs.reset_index(drop=True)
