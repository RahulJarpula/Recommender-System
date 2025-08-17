import pandas as pd
import numpy as np

# Load required files
ratings = pd.read_csv("data/ratings.csv")
courses = pd.read_csv("data/course_genre.csv")
course_embs = pd.read_csv("data/course_bert_emb.csv").values  # shape: (n_courses, dim)

# 1. Map COURSE_ID to its embedding row index
courses = courses.reset_index(drop=True)
course_id_to_index = {cid: idx for idx, cid in enumerate(courses["COURSE_ID"])}  

# 2. Create user → list of embedding vectors
user_embs = {}
for user, user_df in ratings.groupby("user"):
    user_courses = user_df["item"].values  # course IDs
    valid_embs = []
    ratings_list = [] #take weighted mean instead of mean -> cuz rating is in numerical state;
    rating_dict = dict(zip(user_df["item"], user_df["rating"]))

    for cid in user_courses:
        idx = course_id_to_index.get(cid)
        if idx is not None:
            valid_embs.append(course_embs[idx])
            rating = rating_dict.get(cid)
            if rating is not None:
                ratings_list.append(rating)
            
    if valid_embs:
        user_embs[user] = np.average(valid_embs, axis=0, weights=ratings_list)

# 3. Save to CSV
user_ids = []
emb_list = []

for uid, emb in user_embs.items():
    user_ids.append(uid)
    emb_list.append(emb)

user_emb_df = pd.DataFrame(emb_list)
user_emb_df.insert(0, "user", user_ids)
user_emb_df.to_csv("data/user_bert_emb.csv", index=False)

print(f"✅ Saved user embeddings for {len(user_embs)} users.")
