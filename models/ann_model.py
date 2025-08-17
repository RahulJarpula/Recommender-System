import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

# ---------------------
# 1️⃣ RecommenderNet class
# ---------------------
class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_items, embedding_size=16):
        super(RecommenderNet, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        self.user_bias      = nn.Embedding(num_users, 1)
        self.item_bias      = nn.Embedding(num_items, 1)

        nn.init.kaiming_normal_(self.user_embedding.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.item_embedding.weight, nonlinearity='relu')

    def forward(self, user_idx, item_idx):
        u_b   = self.user_bias(user_idx).squeeze()
        i_b   = self.item_bias(item_idx).squeeze()
        u_vec = self.user_embedding(user_idx)
        i_vec = self.item_embedding(item_idx)
        dot   = (u_vec * i_vec).sum(dim=1)
        x     = dot + u_b + i_b
        return torch.relu(x)

# ---------------------
# 2️⃣ Canonicalized Training Function
# ---------------------
def train_ann(rating_path: str,
              epochs: int = 5,
              batch_size: int = 256,
              lr: float = 1e-3,
              embed_dim: int = 16,
              device: str = 'cpu') -> None:
    # Load ratings and clean item IDs
    df = pd.read_csv(rating_path)
    df['item'] = df['item'].astype(str).str.strip()  # Canonical clean

    df['user_idx'] = pd.Categorical(df['user']).codes
    df['item_idx'] = pd.Categorical(df['item']).codes

    num_users = df['user_idx'].nunique()
    num_items = df['item_idx'].nunique()

    # Prepare tensors
    users   = torch.tensor(df['user_idx'].values, dtype=torch.long)
    items   = torch.tensor(df['item_idx'].values, dtype=torch.long)
    ratings = torch.tensor(df['rating'].values, dtype=torch.float32)

    ds = TensorDataset(users, items, ratings)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # Model
    model = RecommenderNet(num_users, num_items, embedding_size=embed_dim).to(device)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for u_batch, i_batch, r_batch in dl:
            u_batch, i_batch, r_batch = (t.to(device) for t in (u_batch, i_batch, r_batch))
            opt.zero_grad()
            preds = model(u_batch, i_batch)
            loss  = loss_fn(preds, r_batch)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"[ANN] Epoch {epoch}/{epochs} — loss={total_loss / len(dl):.4f}")

    # Save model and mappings with canonical IDs
    os.makedirs("models", exist_ok=True)
    torch.save({
        'state_dict': model.state_dict(),
        'user_categories': df[['user', 'user_idx']].drop_duplicates().set_index('user')['user_idx'].to_dict(),
        'item_categories': df[['item', 'item_idx']].drop_duplicates().set_index('item')['item_idx'].to_dict(),
        'embed_dim': embed_dim
    }, "models/ann_model.pkl")
    print("Saved ann_model.pkl to models/")

# ---------------------
# 3️⃣ Canonicalized Prediction Function
# ---------------------
def predict_ann(user_id: int,
                rating_path: str,
                top_n: int = 10,
                device: str = 'cpu') -> pd.Series:
    # Load model and mappings
    artifact = torch.load("models/ann_model.pkl", map_location=device)
    user_map = artifact['user_categories']
    item_map = artifact['item_categories']
    inv_item_map = {v: k for k, v in item_map.items()}

    if user_id not in user_map:
        raise KeyError(f"User {user_id} not in training set")

    num_users = len(user_map)
    num_items = len(item_map)
    embed_dim = artifact['embed_dim']

    model = RecommenderNet(num_users, num_items, embedding_size=embed_dim).to(device)
    model.load_state_dict(artifact['state_dict'])
    model.eval()

    # Predict scores for all items
    u_idx = torch.tensor([user_map[user_id]] * num_items, dtype=torch.long).to(device)
    i_idx = torch.tensor(list(range(num_items)), dtype=torch.long).to(device)

    with torch.no_grad():
        scores = model(u_idx, i_idx).cpu().numpy()

    # Load and clean ratings
    df = pd.read_csv(rating_path)
    df['item'] = df['item'].astype(str).str.strip()

    seen = set(df[df['user'] == user_id]['item'])

    candidates = [
        (inv_item_map[i], score)
        for i, score in enumerate(scores)
        if inv_item_map[i] not in seen
    ]

    print(f"[DEBUG] Seen items: {list(seen)[:5]}")
    print(f"[DEBUG] # of unseen candidates: {len(candidates)}")

    if not candidates:
        print(f"[DEBUG] No unseen candidates for user {user_id}. Returning empty list.")
        return pd.Series([], name='item')

    candidates.sort(key=lambda x: -x[1])
    top_items = [item for item, _ in candidates[:top_n]]

    print(f"[DEBUG] Top predicted items: {top_items}")
    print(f"[DEBUG] Score range: {np.min(scores)} to {np.max(scores)}")

    return pd.Series(top_items, name='item')
