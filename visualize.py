from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

def tsne_plot_with_recommendations(
    embedding_df: pd.DataFrame,
    recommended_items: list,
    save_path: str,
    title: str = "User-Specific t-SNE",
    perplexity: int = 30,
    random_state: int = 42
):
    if embedding_df.empty or len(embedding_df) < 3:
        print("❌ Not enough items to perform t-SNE")
        return

    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(embedding_df) - 1),
        init="pca",
        random_state=random_state
    )
    reduced = tsne.fit_transform(embedding_df.values)
    coords_df = pd.DataFrame(reduced, index=embedding_df.index, columns=["x", "y"])

    plt.figure(figsize=(7, 5))

    # Plot all items (faint background)
    plt.scatter(
        coords_df["x"],
        coords_df["y"],
        s=8,
        alpha=0.2,
        color='gray',
        label="All items"
    )

    # Plot recommended items
    rec_coords = coords_df.loc[coords_df.index.intersection(recommended_items)]
    if not rec_coords.empty:
        plt.scatter(
            rec_coords["x"],
            rec_coords["y"],
            s=50,
            c='red',
            marker='D',
            edgecolors='black',
            label="Recommended items"
        )
        for i, txt in enumerate(rec_coords.index):
            plt.annotate(txt, (rec_coords.iloc[i]["x"], rec_coords.iloc[i]["y"]), fontsize=8)

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
