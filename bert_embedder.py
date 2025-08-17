import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def get_bert_embeddings(course_texts, model_name='all-MiniLM-L6-v2'):
    """
    Generate sentence-level BERT embeddings for a list of course texts.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(course_texts, show_progress_bar=True)
    return embeddings

if __name__ == "__main__":
    # Load your course texts
    courses_df = pd.read_csv("data/course_genre.csv")
    course_ids = courses_df["COURSE_ID"].tolist()
    course_texts = courses_df["TITLE"] 

    # Generate embeddings
    embeddings = get_bert_embeddings(course_texts)

    # Save as .npy or CSV
    np.save("data/course_bert_emb.npy", embeddings)
    df = pd.DataFrame(embeddings, index=course_ids)
    df.index.name = "item"  # So when saved, first column is 'item'
    # Save with course IDs as index
    df.to_csv("data/course_bert_emb.csv")
    print(f"✅ Saved BERT embeddings for {len(course_texts)} courses.")
