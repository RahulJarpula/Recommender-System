import pandas as pd
import numpy as np

ratings_df = pd.read_csv("data/ratings.csv")

users = ratings_df['user'].unique()
np.random.shuffle(users)
split_idx = int(0.8 * len(users))
train_users = users[:split_idx]
test_users = users[split_idx:]

train_df = ratings_df[ratings_df['user'].isin(train_users)]
test_df  = ratings_df[ratings_df['user'].isin(test_users)]

train_df.to_csv("data/train_ratings.csv", index=False)
test_df.to_csv("data/test_ratings.csv", index=False)
