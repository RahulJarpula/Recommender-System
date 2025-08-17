# recommender_app.py
import numpy as np
import os
import streamlit as st
import pandas as pd
import backend
from collections import Counter
from evaluate import train_test_split_ratings 
from visualize import tsne_plot_with_recommendations
from backend import predict
from datetime import datetime

st.set_page_config("Course Recommender", layout="wide")
st.title("📚 Course Recommendation Demo")
from models.knn_model       import train_knn,       predict_knn
from models.nmf_model       import train_nmf,       predict_nmf
from models.class_emb_model import train_class_emb, predict_class_emb
from models.reg_emb_model   import train_reg_emb,   predict_reg_emb
from models.ann_model       import train_ann,       predict_ann

# 1. Model selector
model = st.sidebar.selectbox("Choose Model", list(backend.MODELS))
embed_type = st.sidebar.radio("Embedding Type", ["BoW", "BERT"])

MODELS = {
    "KNN":       (train_knn,       predict_knn),
    "NMF":       (train_nmf,       predict_nmf),
    "ClassEmbd": (train_class_emb, predict_class_emb),
    "RegEmbd":   (train_reg_emb,   predict_reg_emb),
    "ANN":       (train_ann,       predict_ann),
}

# 2. (Optional) Load common data for EDA
@st.cache_data
def load_ratings():
    return pd.read_csv("data/ratings.csv")

#ratings_df, user_emb_df, item_emb_df = backend.load_data()
ratings_df = pd.read_csv("data/ratings.csv")
train_df, test_df, test_users, user_emb_df, item_emb_df = backend.load_data(embed_type=embed_type)
valid_users = test_df[test_df["rating"] >= 4]["user"].unique().tolist()
valid_users.sort()


# 3. Hyper-parameter widgets
params = {}
if model == "KNN":
    params["ratings_df"]  = ratings_df
    params["n_neighbors"] = st.sidebar.slider("k (neighbors)", 1, 50, 10)
    params["metric"]      = st.sidebar.selectbox("Similarity Metric", ["cosine","pearson","msd"])
    params["user_based"]  = st.sidebar.checkbox("User-based CF", False)

elif model == "NMF":
    params["rating_path"] = "data/ratings.csv"
    params["n_factors"]   = st.sidebar.slider("Latent Factors", 10, 100, 50)
    params["n_epochs"]    = st.sidebar.slider("Epochs", 5, 50, 15)

elif model == "ClassEmbd":
    user_emb_file  = "data/user_bert_emb.csv" if embed_type == "BERT" else "data/user_emb.csv"
    item_emb_file  = "data/course_bert_emb.csv" if embed_type == "BERT" else "data/item_emb.csv"
    params.update({
      "rating_path":     "data/ratings.csv",
      "user_emb_path":   "data/user_emb.csv",
      "item_emb_path":   "data/item_emb.csv",
      "model_type":      st.sidebar.selectbox("Classifier", ["logistic","random_forest"]),
      "test_size":       st.sidebar.slider("Test Size", 0.1, 0.5, 0.2),
      "random_state":    42
    })

elif model == "RegEmbd":
    user_emb_file  = "data/user_bert_emb.csv" if embed_type == "BERT" else "data/user_emb.csv"
    item_emb_file  = "data/course_bert_emb.csv" if embed_type == "BERT" else "data/item_emb.csv"
    params.update({
      "rating_path":     "data/ratings.csv",
      "user_emb_path":   "data/user_emb.csv",
      "item_emb_path":   "data/item_emb.csv",
      "test_size":       st.sidebar.slider("Test Size", 0.1, 0.5, 0.2),
      "random_state":    42
    })

elif model == "ANN":
    user_emb_file  = "data/user_bert_emb.csv" if embed_type == "BERT" else "data/user_emb.csv"
    item_emb_file  = "data/course_bert_emb.csv" if embed_type == "BERT" else "data/item_emb.csv"
    params.update({
      "rating_path":     "data/ratings.csv",
      "user_emb_path":   "data/user_emb.csv",
      "item_emb_path":   "data/item_emb.csv",
      "epochs":          st.sidebar.slider("Epochs", 5, 50, 15),
      "batch_size":      st.sidebar.number_input("Batch Size", 32, 1024, 256),
      "lr":              st.sidebar.number_input("Learning Rate", 1e-4, 1e-2, 1e-3, format="%.4f"),
      "embed_dim":       st.sidebar.slider("Embed Dim", 8, 128, 64),
      "device":          "cpu"
    })

# 4. Train button
with st.sidebar:
    if st.button("Train"):
        with st.spinner("Training model..."):
            # Load the data needed for training
            train_df, test_df, test_users, user_emb_df, item_emb_df = backend.load_data(embed_type=embed_type)
            # Find first user in test set with at least 1 item rated ≥ 4
            valid_user_id = None
            for uid in test_df["user"].unique():
                user_items = test_df[(test_df["user"] == uid) & (test_df["rating"] >= 4)]
                if len(user_items) > 0:
                    valid_user_id = uid
                    break

            if valid_user_id is None:
                raise ValueError("❌ No user in test set has rating ≥ 4")

            print(f"✅ Using valid user: {valid_user_id}")
            # print("🔍 test_df sanity check (top 5 rows):")
            # print(test_df.head())
            # print("🔍 test_df.columns:", test_df.columns.tolist())

            # Inject into params for models that need it
            if model == "KNN" or model == "NMF":
                params["ratings_df"] = ratings_df
            backend.train(model, **params)
        st.sidebar.success("✔ Model trained!")

# 5. Recommendation interface
user_id = st.selectbox("Choose a user (in test set)", valid_users)
top_k   = st.sidebar.number_input("Top K", min_value=1, max_value=20, value=10)

if st.sidebar.button("Recommend") and user_id:
    with st.spinner("Generating recommendations…"):
        if model in {"KNN", "NMF"}:
            params["rating_path"] = "data/ratings.csv"
        recs = backend.predict(model, user_id=user_id, top_n=top_k, **params)
    st.subheader(f"Top {top_k} recommendations for {user_id}")
    st.table(recs)
    #st.write("🔍 Debug: Recommendations Output")
    #st.write(recs)

    #if model in ["ANN", "RegEmbd", "ClassEmbd", "KNN", "NMF"]:
        # with st.expander("📊 Evaluation Metrics"):
        #     try:
        #         course_vecs_dict = {
        #             course_id: vector
        #             for course_id, vector in zip(item_emb_df["item"], item_emb_df.iloc[:, 1:].values)
        #         }

        #         popularity_dict = dict(Counter(ratings_df["item"]))  # still use full set for popularity
        #         all_items = ratings_df["item"].unique()
        #         st.write("🧪 Columns in test_df:", test_df.columns.tolist())
        #         # ✅ USE test_df HERE
        #         metrics = evaluate_model(
        #             model_name=model,
        #             user_ids=[user_id],
        #             top_n=top_k,
        #             train_df=train_df,
        #             test_df=test_df,
        #             item_embeddings_df=item_emb_df.set_index("item"),
        #             item_popularity=dict(Counter(train_df["item"])),
        #             embed_type="bert" if "bert" in model.lower() else "bow"
        #         )



        #         for name, value in metrics.items():
        #             st.write(f"**{name.capitalize()}@{top_k}:** {value:.4f}")

        #     except Exception as e:
        #         st.warning(f"Evaluation error: {e}")

# t-SNE Visualization Block
# --- Model Comparison ---
st.markdown("### 🔄 Before vs After: Embedding Comparison")

# --- User Selection ---
#user_ids = pd.read_csv("data/test_ratings.csv")["user"].unique().tolist()
user_id = st.selectbox("Choose a user to inspect:", valid_users)

# --- Side-by-side Model + Embedding Selectors ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### BoW-based Recommender")
    model_a = st.selectbox("Model A", list(MODELS.keys()), key="bow_model")
    embed_a = "bow"  # fixed

with col2:
    st.markdown("#### BERT-based Recommender")
    model_b = st.selectbox("Model B", list(MODELS.keys()), key="bert_model")
    embed_b = "bert"  # fixed

# --- Compare Button ---
if st.button("Compare BoW vs BERT Recommendations"):
    
    def predict_with_embed_type(model, user_id, embed_type):
        rating_path = "data/train_ratings.csv"
        user_emb_path = "data/user_emb.csv" if embed_type == "bow" else "data/user_bert_emb.csv"
        item_emb_path = "data/item_emb.csv" if embed_type == "bow" else "data/course_bert_emb.csv"

        if model == "KNN":
            return predict(
                model_name=model,
                user_id=user_id,
                top_n=5  # no rating_path needed
            )

        elif model == "NMF":
            return predict(
                model_name=model,
                user_id=user_id,
                top_n=5,
                #rating_path=rating_path  # gets renamed to ratings_df in backend
            )

        elif model in ["ClassEmbd", "RegEmbd"]:
            return predict(
                model_name=model,
                user_id=user_id,
                top_n=5,
                rating_path=rating_path,
                user_emb_path=user_emb_path,
                item_emb_path=item_emb_path
            )

        elif model == "ANN":
            return predict(
                model_name=model,
                user_id=user_id,
                top_n=5,
                rating_path=rating_path,
                device="cpu"
            )

        else:
            raise ValueError(f"Unsupported model: {model}")

    # Generate recs using backend.py
    recs_a = predict_with_embed_type(model_a, user_id, embed_type="bow")
    recs_b = predict_with_embed_type(model_b, user_id, embed_type="bert")


    col1.dataframe(pd.DataFrame({"BoW Recommendations": recs_a}))
    col2.dataframe(pd.DataFrame({"BERT Recommendations": recs_b}))

    # --- t-SNE Plot for Each ---
    item_emb_df = pd.read_csv("data/item_emb.csv", index_col=0)
    tsne_plot_with_recommendations(item_emb_df, recs_a, "bow_tsne.png", "BoW t-SNE (Top-5)")

    bert_emb_df = pd.read_csv("data/course_bert_emb.csv", index_col=0)
    tsne_plot_with_recommendations(bert_emb_df, recs_b, "bert_tsne.png", "BERT t-SNE (Top-5)")

    st.image("bow_tsne.png", caption="BoW t-SNE (Recommended Items)")
    st.image("bert_tsne.png", caption="BERT t-SNE (Recommended Items)")

# --- Feedback ---
st.markdown("### ✍️ Give Feedback")

col1, col2 = st.columns(2)
use_feedback = st.sidebar.checkbox("Use Feedback-Based Reranking", value=True)
params["use_feedback"] = use_feedback

with col1:
    feedback_a = st.radio("Were BoW-based recs useful?", ["👍 Yes", "👎 No"], key="fa")
with col2:
    feedback_b = st.radio("Were BERT-based recs useful?", ["👍 Yes", "👎 No"], key="fb")

log_path = "data/user_feedback_log.csv"
write_header = not os.path.exists(log_path) or os.stat(log_path).st_size == 0
if st.button("Submit Feedback"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a") as f:
        if write_header:
            f.write("user_id,item_id,model_type,feedback,timestamp\n")
        f.write(f"{user_id},{model_a},bow,{feedback_a},{now}\n")
        f.write(f"{user_id},{model_b},bert,{feedback_b},{now}\n")
    st.success("Thank you for your feedback!")

