# Module: Text Cleaning & EDA for a Course Recommender (Stopwords, WordCloud, and Popularity Insights)

I kicked off this module after realizing our raw catalog and ratings were noisy and opaque: items were cryptic IDs, course blurbs were stuffed with filler words, and I had no quick feel for which genres dominated the catalog or how active users really were. The goal was twofold: (1) clean and vectorize course text so later models (BoW/TF-IDF or embeddings) learn useful signals, and (2) profile interaction data to surface popularity patterns that would guide baselines and UI choices.

I started by **joining human-readable titles** into the ratings using a left merge on `ratings_df.item → course_df.COURSE_ID`. With titles in place, I profiled interactions: `ratings_df.groupby('user').size()` gave me a **histogram of ratings per user**, revealing the classic long-tail; `ratings_df['item'].value_counts()` exposed **top-rated items**. For **genre popularity**, I summed one-hot columns in `course_df.iloc[:, 2:].sum()` and sorted them descending. When I hit a cryptic `UFuncTypeError` sorting mixed dtypes, I traced it to accidentally using NumPy’s `sort` on a list of `(str, Int64Dtype)` tuples. I fixed it by switching to **Python’s built-in `sorted()`** (or pandas’ `Series.sort_values()`), and also avoided shadowing built-ins like `dict`.

On the text side, I established a **cleaning pipeline**: lowercasing → punctuation handling / tokenization (word-level) → **stopword filtering** (NLTK) → **lemmatization** → optional **phrase handling** (hyphens, contractions). I curated a **domain stopword list** (e.g., “course”, “learn”, “introduction”, “module”, “session”) so the model focuses on subject matter rather than marketing fluff. Using **Gensim**, I built a `Dictionary` and converted each description via `doc2bow`, then layered a **TF-IDF model** to reweight distinctive terms. To communicate results quickly, I generated a **WordCloud** from cleaned tokens (custom stopwords, controlled `max_words`, white background) and paired it with **seaborn bar plots** (top genres, top items) with readable axes and annotations.

There were a few gotchas: Boolean masking required `&` (not `and`), and groupby objects don’t sort until you **materialize a Series** (`size()`, `value_counts()`). I added defensive steps—`dropna()`, explicit `astype(str)`—so joins and text joins never crash on unexpected types.

**Impact:** This module gave us (a) **clean, compact text features** that improved downstream model stability, (b) **clear visuals** (WordCloud, histograms, bar charts) that sped up stakeholder discussions, and (c) **actionable catalog insights** (genre skew, item popularity) to seed baselines and UI defaults. It also laid a **future-proof path** for swapping BoW/TF-IDF with embeddings.

**Keywords & skills:** pandas (`merge`, `groupby`, `size`, `value_counts`, `sort_values`), matplotlib/seaborn, NLTK stopwords, lemmatization, Gensim `Dictionary`/`doc2bow`/TF-IDF, WordCloud, data hygiene (dtypes, NaNs), boolean masks, error debugging (`UFuncTypeError`, built-in shadowing).

**Summary:** By turning messy text and IDs into clean features and intuitive visuals, this module transformed our dataset into something models—and humans—can reason about, accelerating the recommender’s path from exploration to reliable baselines.

# Module 1 — Tokenize → Bag-of-Words → Course Similarity (Content-Based)

I started this module with a simple pain point: our app couldn’t **explain** or **bootstrap** recommendations for new or lightly-rated courses. Collaborative signals were thin, so I needed a content signal—something I could compute directly from the course text itself and use to answer, “which courses are most alike?” That became the goal: build an interpretable, scalable **content-based similarity engine**.

First, I set up lightweight EDA: loading data with **pandas**, sanity-checking schemas (`COURSE_ID`, titles, genre flags), and a quick **WordCloud** after removing **stopwords** to see dominant terms. This validated that titles/descriptions actually contained useful signal (“machine learning”, “beginners”, “data science”). Then I implemented two parallel tokenization→vectorization paths:

1. **Gensim Dictionary route**:
   I tokenized text (lowercasing, basic punctuation handling via **NLTK** tokenizers), built a `corpora.Dictionary` over the corpus, and converted each course into sparse BoW using `doc2bow`. For analyses that wanted dense matrices, I expanded the sparse tuples into a **wide** DataFrame: **rows = courses**, **columns = vocabulary tokens**, **values = counts**.

2. **TorchText vocab route**:
   To keep the pipeline framework-agnostic, I also built a vocab using `build_vocab_from_iterator`, set `<unk>` for OOV safety, and wrote a tiny `bow_vector()` to increment counts per token. This path let me produce BoW directly as **PyTorch** tensors, stack them (`(n_courses, vocab_size)`), and drop them into any downstream model without a pivot.

With vectors in place, I computed pairwise similarities using **scikit-learn**: **cosine** (directional alignment), **Euclidean** (absolute differences), and **Jaccard** on binarized counts (set overlap). I wrapped the results in a similarity matrix/DataFrame for easy lookup and built a neat “top-k similar courses” helper per course ID.

**Challenges & fixes:**

* **Schema joins**: aligning `ratings_df.item` ↔ `course_df.COURSE_ID` (fixed with a left merge to attach titles for readability).
* **Orientation confusion** (tokens as rows vs. columns): standardized on **courses-as-rows**, which made pairwise similarity trivial.
* **Jaccard correctness**: remembered to **binarize** counts before applying it.
* **OOV handling**: `<unk>` in the TorchText path; Gensim naturally ignores unseen tokens.
* **Scale**: for larger corpora, I planned top-k retrieval instead of full $n\times n$ similarity to keep latency low.

**Impact:**

* Immediate **cold-start coverage** for new items (no ratings required).
* **Explainability**: we can show overlapping tokens/topics for any recommendation.
* **Developer velocity**: two interchangeable BoW backends (Gensim or TorchText) and clean pandas utilities made iteration fast.
* **Scalability path**: sparse storage + top-k retrieval readies the system for growth.

**Keywords & skills:** pandas, NLTK, Gensim `Dictionary`/`doc2bow`, TorchText `build_vocab_from_iterator`, Bag-of-Words, stopwords, WordCloud, cosine/Jaccard/Euclidean, pairwise similarity, DataFrame pivot/stack, PyTorch tensors, explainability.

**Summary:** This module turned raw course text into actionable vectors and a transparent similarity engine. It laid the foundation for cold-start recommendations, faster iteration, and user-friendly explanations—an essential building block for the rest of the recommender.

Here’s how I told the story of **Module 2: Content-Based User Profiling → PCA → K-Means → Cluster-Driven Recommendations**.

I began with a simple question: *how do we recommend the right courses when all we have are user ratings and course attributes?* Our earlier baseline produced lists, but it wasn’t scalable or interpretable. So I designed a pipeline that (a) **learns a taste profile for each user**, (b) **compresses those profiles** into a compact space, and (c) **segments users** into actionable groups to drive recommendations—even for brand-new courses.

**Step-by-step implementation.** I built **user taste vectors** by multiplying each user’s rating vector with the **course–genre matrix** (content-based filtering via matrix dot products). Using **pandas/NumPy**, I kept everything vectorized. I standardized features with **StandardScaler** to prevent any single genre from dominating. Then I fit **PCA** and chose the dimensionality by the **90% cumulative explained variance** rule (verified with a scree/bar plot). On the reduced embeddings, I ran **KMeans**, picking **k** via the **elbow method** and validating with **silhouette score**. Finally, I created a **cluster-popularity recommender**: for a given user, find their cluster, identify courses popular within that cluster (via `groupby` counts and a threshold), subtract what the user already took (set difference), and recommend the **unseen popular** items. I also added quick utilities: `dict(zip(doc_id, doc_index))` for **course-to-index mapping**, careful `.iloc` slicing for features, and **reproducible** `random_state` seeds.

**Challenges & fixes.** Early on, I accidentally mixed a **token vocabulary** (BoW) with **course indices**, causing `KeyError`s; I separated **`course_idx_map`** from **`bow_vocab`**. Another pitfall: losing the `'user'` column by overwriting the raw table with an aggregated one; I kept a dedicated `raw` DataFrame and wrote grouped results to a new variable. Picking k naively by minimum inertia overfit; switching to **elbow + silhouette** solved that. Finally, I ensured the PCA plot used the right axis (`X.shape[1]`) and saved `fit_transform` outputs before concatenating with IDs.

**Impact.** Inference became **faster** (smaller PCA space → quicker KMeans and scoring), **more interpretable** (cluster “personas” with centroid profiles), and **more scalable** (vectorized ops). The **cluster-popularity stage** boosted **coverage/diversity** while keeping relevance high, and the pipeline enabled **cold-start for new courses** by projecting them through the same **scaler → PCA → KMeans** to reach the right audience immediately.

**Keywords/skills:** pandas, NumPy, scikit-learn, StandardScaler, PCA (explained variance), KMeans (elbow, silhouette), cosine/similarity matrix, content-based filtering, dot product, `groupby`, set operations, vectorization, cold-start, reproducibility.

**Summary:** Module 2 transformed raw ratings into **compact user embeddings**, uncovered **coherent user clusters**, and delivered a **practical, scalable recommender** that balances accuracy with diversity—ready to drive real recommendations and absorb new courses on day one.


### Final Module — Streamlit Front-End + Multi-Model Backend (KNN, NMF, ANN, Reg/Clf on Embeddings)

When I reached the last mile of the recommender project, the real problem wasn’t just “which model performs best?”—it was **how to make the system usable, reproducible, and demo-ready**. I needed a single interface that let me train different recommenders, tweak hyper-parameters on the fly, visualize the data, and generate recommendations for any user. This final module turned the scattered notebooks into a cohesive **app** with a disciplined **backend**.

**How I built it:** I started by extracting clean `train_*` and `predict_*` functions from five notebooks—**KNN** and **NMF** via `surprise`, **ANN** via **PyTorch**, and **regression/classification on embeddings** via **scikit-learn**. Each model now serializes to its own `.pkl` file for fast re-use. I standardized signatures and created a central **dispatcher** in `backend.py`:

```python
MODELS = {"KNN":(train_knn,predict_knn), "NMF":(train_nmf,predict_nmf),
          "ANN":(train_ann,predict_ann), "RegEmbd":(train_reg_emb,predict_reg_emb),
          "ClassEmbd":(train_class_emb,predict_class_emb)}
```

This gave me a uniform `backend.train(name, **params)` / `backend.predict(name, **params)` API. On the front-end, I built **`recommender_app.py`** with **Streamlit**, adding sidebar controls for model choice and hyper-parameters, plus buttons for **Train** and **Recommend**. I introduced **expanders** for **EDA** (rating distributions, top users/items) and **Model Metrics** (hook for RMSE/MAE for Surprise models, accuracy/F1 for classification, MSE/R² for regression). To keep the app responsive, I used `@st.cache_data` and `@st.cache_resource` where appropriate, and vectorized inference paths (e.g., tiling user vectors and stacking with item embeddings in NumPy for fast batch scoring).

**Challenges & fixes:** Installing `scikit-surprise` on macOS with Python 3.13 failed due to Cython/OpenMP. I solved this by isolating a **Python 3.10 venv** (`venv_py310`) specifically for Surprise, while keeping the rest of the stack intact—clean environment management made the build reliable. I also fixed dependency gaps (`torch`, `pandas`) across venvs, standardized file paths (`data/ratings.csv`, `data/user_emb.csv`, `data/item_emb.csv`), and implemented robust **ID↔index mapping** (like NLP vocabularies) to support embedding layers and Surprise’s inner/outer IDs.

**Impact:**

* **Velocity:** Switching models became a dropdown action; training and inference dropped from minutes of manual notebook work to a few clicks.
* **Reliability & Reproducibility:** Versioned `.pkl` artifacts and a single dispatcher API reduced glue code and errors.
* **UX & Trust:** EDA and metrics made behavior explainable; stakeholders could see **why** items were suggested and how models performed.
* **Scalability of workflow:** Modular design let me add or swap algorithms without touching the UI.

**Keywords:** Streamlit, PyTorch, scikit-surprise (KNN, NMF), scikit-learn (LinearRegression, LogisticRegression, RandomForest), pandas, NumPy, embeddings, serialization (`pickle`), caching, RMSE/MAE, MSE/R², F1, EDA, ID→index mapping, dispatcher pattern, virtual environments.

**Summary:** This module transformed a set of experimental notebooks into a polished, **multi-model recommender application**—interactive, explainable, and production-lean—ready to demo and easy to extend.

Here’s the story of how I built and hardened the **Evaluation & Diagnostics module** of my recommender system—the piece that turns raw predictions into trustworthy, resume-grade metrics.

I began with an obvious pain point: my app showed clean recommendations, but the **metrics were wrong or always zero**. Precision\@K and Recall\@K flatlined, novelty looked nonsensical, and I couldn’t tell if the model was genuinely bad or if my evaluation was. The first step was to **stabilize the data path**. I centralized loading in `backend.load_data()` to read `ratings.csv`, perform a **user-wise train/test split** (`train_test_split_ratings`), and load embeddings (BoW or BERT) via deterministic file paths. This eliminated hidden state and ensured **reproducible evaluation**.

Next, I added **schema guards and debug probes**. Before grouping by user, I printed columns and head rows, asserted `["user","item","rating"]` in `test_df`, and surfaced errors early. This immediately exposed a subtle bug: I had mistakenly **passed the item-embedding matrix as `test_df`** due to **misaligned positional arguments**. I fixed the call by switching to **explicit keyword arguments** in `evaluate_model(model_name, user_ids, top_n, train_df, test_df, item_embeddings_df, item_popularity, embed_type)`—no more silent parameter shifts.

With clean data flowing, I corrected the **novelty** metric. Instead of logging an undefined `recs` list and using raw log(popularity), I implemented the proper definition: **average −log₂(popularity)** over the top-K across all users, with safe defaults for unseen items. I also enforced **masking of seen items** from `train_df` so recommendations don’t “cheat.”

A big UX issue was users with no relevant ground truth, which guaranteed Precision/Recall=0. I solved this at the UI layer: in Streamlit, I **filter the user dropdown to only those in `test_df` with rating ≥ 4**, and I moved this logic **outside** the “Train” button so it persists. I also added an optional **batch evaluator** to score **multiple valid users**, enabling **Inter-list Diversity** and fairer averages.

Challenges included debugging dataframe shape mismatches (embeddings vs. ratings), guarding against **data leakage**, and keeping the evaluation fast. I used **Python 3.10**, **pandas**, **NumPy**, **Streamlit**, and model backends (KNN/NMF, cosine similarity, BoW/BERT via SentenceTransformers). Optimizations included **caching loads**, **indexing embeddings by `item`**, and **keyword-only evaluation calls**.

Impact: metrics are now **stable, interpretable, and reproducible**. Precision/Recall actually reflect model quality, **Novelty** is meaningful, **Intra/Inter-list Diversity** surfaces diversity trade-offs, and **Catalog Coverage** tells me how much of the catalog I’m using. From a user’s perspective, the app no longer allows invalid user selections, reducing confusion and evaluation noise.

**Keywords:** Python 3.10, pandas, NumPy, Streamlit, train/test split (user-wise), data leakage prevention, BERT & BoW embeddings, cosine similarity, KNN, NMF, Precision\@K, Recall\@K, Novelty, Intra/Inter-list Diversity, Catalog Coverage, schema validation, keyword arguments.

**Summary:** This module transformed my recommender from “looks fine” to **measurably correct**—a robust evaluation pipeline that surfaces real strengths and weaknesses, guides model iteration, and elevates the project to production-ready quality.

Here’s the story of how I built the **Feedback-Aware Re-ranking module** that let the recommender “learn” from users in real time.

I started with a clear problem: traditional offline metrics (Precision\@k, Recall\@k) weren’t telling a useful story for my system, and inter-list diversity was flat. I needed a *human-in-the-loop* signal to correct the ranking, prove adaptivity in demos/interviews, and make the Streamlit app feel alive. That became the goal for this module: **collect user feedback seamlessly and use it to re-score future recommendations.**

I began in the UI. In `recommender_app.py`, I added a **feedback panel** under the results: thumb-up/down radios for each model, a toggle to enable “Use Feedback-Based Reranking,” and a submit button that appends to `data/user_feedback_log.csv`. To make the log robust, I wrote a small header-guard that writes `user_id,item_id,model_type,feedback,timestamp` when the file is new. I also added a defensive rename (if a legacy column wasn’t called “feedback”, rename the 4th column to `feedback`) and a quick debug print to surface schema issues. A “Most Liked Courses” expander charts aggregated likes (mean of 👍/👎 mapped to 1/0) and a simple “Most Feedback Received” view to expose catalog hot spots.

Next, I pushed the learning into the backend. In `backend.py`, I centralized a **post-processing step** so I wouldn’t duplicate logic across models:

* `get_feedback_scores()` loads the CSV, maps 👍/👎 → 1/0, aggregates into an **item favorability score** (with a neutral default of 0.5 for unseen items).
* `feedback_rerank(scores_dict, item_feedback, alpha=0.8, beta=0.2)` computes
  `final = α * similarity + β * feedback`, letting me blend model relevance with crowd preference.
* `postprocess_rerank(preds, use_feedback, top_n)` converts any model output (list or DataFrame with scores) into `{item: score}`, applies feedback, sorts, and returns top-N.
  I call this from **every** model path (KNN/NMF/ANN/RegEmbd/ClassEmbd) so the feature is universal but **toggleable** from the UI.

I faced a few gotchas: a `KeyError: 'feedback'` when older logs had inconsistent headers; I fixed it with the rename + `header=0` read and a schema fallback creating an empty DataFrame if columns don’t match. I also guarded against empty logs, corrupted rows, and added a neutral default so new items were not unfairly penalized. Finally, I exposed optional knobs: **popularity penalty** and **MMR** reranking to balance novelty and redundancy if needed.

Impact-wise, this module transformed the app from a static demo into an **adaptive system**. Product-wise: clearer **interpretability** (“why did this move up?”), better **user trust**, and a crisp **interview narrative** about online learning and human-in-the-loop ML. It also improved catalog coverage and perceived relevance in qualitative tests and provided actionable analytics for curation.

**Keywords/Skills:** Streamlit, Pandas, CSV schema guard, human-in-the-loop, online learning, reranking, late fusion, cosine similarity, TF-IDF/BERT embeddings, MMR (Maximal Marginal Relevance), popularity penalty, data visualization, error handling, production-minded interfaces.

**Summary:** I built a full feedback loop—UI capture → robust logging → centralized reranking—so the recommender continuously adapts to users and tells a convincing, production-ready story.

Here’s the story of the **“Embedding Visualization & User-Controlled Recommender”** module—the piece that turned our engine from a black box into a guided cockpit.

I built this after noticing a real tension: **neural embeddings (BERT) often surfaced a multi-topic mix** (Python + R + Databases) while **BoW/TF-IDF** tended to **collapse into one dominant cluster**. For a learner, both could be “right,” depending on whether they wanted breadth or depth. The project needed **transparency and control**—a way to show how the models “see” the catalog and let the user pick the behavior they prefer.

**Implementation (chronological):**
I started by standardizing two embedding pipelines: **BoW/TF-IDF** (`min_df`, `ngram_range`, L2-normalization) and **BERT/Sentence-BERT** (unit-norm vectors). To make visualization feasible, I applied **PCA → 50D** as a speed/denoising stage. Then I added two projection options:

* **t-SNE** (sklearn): tuned **perplexity**, **learning\_rate**, and **early\_exaggeration**; fixed `random_state` for deterministic runs.
* **UMAP** (umap-learn): used **metric="cosine"**, tuned **n\_neighbors** and **min\_dist** for better topic separation with BERT.

I wrapped these in **Streamlit** with **Plotly** scatterplots: colors by topic/cluster, hover tooltips (title, tags, difficulty), and a legend toggle. A simple **radio switch** let users choose the recommendation source—**BoW**, **BERT**, or **Hybrid (BERT + MMR re-rank)**. Under the hood, I standardized the scoring interface (`predict(user_id, embed_type="bow"|"bert")`) with **cosine similarity** and an optional **MMR** pass to balance relevance and diversity. For responsiveness, I **precomputed embeddings and 2D coordinates** and cached them via `st.cache_data` (and persisted to disk with `joblib`)—so plots render instantly.

To **close the loop**, I added a **feedback logger**: thumbs-up/down and model choice write to `user_feedback_log.csv` with a strict schema (`user_id, model_choice, item_id, feedback, timestamp`). A nasty surprise—**KeyError: 'feedback'**—was caused by inconsistent column names from early logs. I wrote a tiny **migration/normalization** utility that patches historical files and enforces the canonical header before appending.

**Challenges & how I solved them:**

* **t-SNE instability** → fixed seeds, PCA pre-step, and parameter sweeps on a held-out subset.
* **BERT cluster “glue” effect** (everything too close) → switched visualization to **UMAP(cosine)** and tuned neighbor/min-dist; clusters became readable.
* **UI state** across tabs (plots vs. recs) → centralized Streamlit session state and a single source of truth for the selected model.
* **CSV schema drift** → header normalization and input validation before writes.

**Impact:**
Users now **see** why BoW and BERT disagree and can **choose** the slate that fits their learning goal (deep dive vs. exploration). The module improved **trust**, **engagement**, and **catalog coverage awareness**; qualitatively, we observed more deliberate switches between BoW for focus and BERT/Hybrid for discovery. Internally, the plots became a **diagnostic tool**—we spot failure modes quickly and justify re-ranking choices.

**Keywords & skills:** TF-IDF, Sentence-BERT, cosine similarity, k-NN retrieval, **t-SNE**, **UMAP**, **PCA**, **MMR re-ranking**, Streamlit, Plotly, caching, data schema validation, human-in-the-loop.

**Summary:** This module transformed recommendations from opaque lists into **explainable, user-steerable experiences**, elevating the project from a student demo to a **resume-ready, interview-strong** system that blends interpretability with practical control.

### Module: Notebook → Production Pipeline (Backend + Streamlit Integration)

When the project moved beyond exploration, my scattered `.ipynb` notebooks (tokenization, merging, EDA, model training) started to slow me down. Re-running cells in the “right order” became a ritual, and copy-pasting code across notebooks risked silent drift. This module was my inflection point: turn exploratory notebooks into a clean, testable Python package that a single Streamlit app could import and run reliably.

I began by inventorying the notebooks and carving them into cohesive modules: `preprocessing/` (tokenize, clean, merge), `features/` (BoW builders, similarity functions), `clustering/` (cluster-based CF), and `models/` (PyTorch ANN). I used `jupyter nbconvert --to script` as a bootstrap, then hand-refactored into functions with docstrings and type hints. Data code returned DataFrames; model code returned artifacts (embeddings, similarity matrices, trained weights). I wired a thin orchestration layer in `backend.py` that exposed just a few high-level entry points: `load_data()`, `init_vocab_and_bow()`, `get_content_recs()`, `get_cluster_recs()`, `train_ann()`, and `predict_ann()`.

Next, I optimized for runtime. Heavy I/O and model loads were wrapped with Streamlit’s `@st.cache_data` / `@st.cache_resource`, ensuring BoW matrices, vocabularies (from `torchtext`/`sklearn`), and similarity blocks (SciPy/NumPy) were computed once and reused. I standardized sparse representations (`scipy.sparse`) to keep memory predictable and vectorized the similarity routines for batch lookups. For the ANN, I isolated the PyTorch training loop and added deterministic seeds, clear `train/val` splits, and `torch.save` checkpoints. Wherever possible, I pushed plotting to functions that return `matplotlib` Figures so the UI could simply display them.

A few snags surfaced. Notebook-only globals were leaking state; I fixed this by passing explicit params and returning pure values. Dependency drift appeared across notebooks; I consolidated into a single `requirements.txt`, added `pre-commit` (black, ruff), and a small `pytest` suite to lock down regressions. Streamlit reruns initially recomputed everything; caching and idempotent functions solved that. Finally, I abstracted file paths and random seeds into a `settings.py` so local and cloud runs behaved identically.

The impact was immediate: faster app cold-start, snappy interactions (no redundant recomputation), and a single source of truth for each algorithm—Content-Based (cosine/Jaccard/Euclidean over BoW), Cluster-CF (user→cluster→popularity ranking), and ANN rating prediction. Maintenance improved drastically: new features now land as modules, not ad-hoc cells.

**Keywords & skills:** Streamlit, PyTorch, scikit-learn, torchtext, SciPy, `scipy.sparse`, cosine/Jaccard/Euclidean similarity, caching, modularization, `nbconvert`, CI-friendly structure, type hints, black/ruff, pytest, reproducibility.

**In short:** this module transformed messy exploration into a production-ready backbone—clean APIs, cached performance, and a Streamlit UI that reliably showcases every model without the notebook overhead.

Here’s how I built **Module 3 – Neural Recommender with Embeddings (PyTorch)**, and why it mattered.

I’d already prototyped neighborhood models (Surprise KNN) and factorization (NMF), but I needed a learner that could **jointly model user–item interactions** and flex with future features. That’s where a small **ANN with embeddings** fit perfectly: it compresses each user and item into a learnable vector in the same latent space, and predicts ratings via a simple **dot product + bias**—fast, expressive, and easy to tune.

I began by **encoding IDs**. Using `LabelEncoder`, I mapped raw `user`/`item` keys to contiguous indices—exactly what an `nn.Embedding` expects. I **scaled ratings** to `[0,1]` with `MinMaxScaler` for stable training, then created clean splits with `train_test_split`: 80% train, 10% val, 10% test. The model itself is compact: two embedding tables (`nn.Embedding`) for users and items, learned **per-user** and **per-item biases**, and a `forward` that returns `ReLU(dot(u, i) + b_u + b_i)`. I optimized with **Adam** on **MSE**, tracking **RMSE** epoch-by-epoch. For data feeding I supported both **DataLoader** (batched, shuffled) and a lightweight `zip(X, y)` loop (handy for small runs). I wrote a tiny `evaluate(X, y, model, criterion)` helper that switches to eval mode, disables grads, aggregates loss across the split, and returns RMSE—essentially a PyTorch version of Keras’s `.evaluate()`.

I hit a few bumps and fixed them methodically. **CUDA library errors** in the environment? I enforced CPU-only training and removed `.to(device)` clutter. A **1D scaler error**? I reshaped ratings to `(-1, 1)` before scaling. A `'numpy.ndarray' object has no attribute values'` bug? I stopped calling `.values` on arrays and either stayed in pandas or NumPy end-to-end. A `NameError` from zipping only features? I zipped **both** `X` and `y`. Finally, I added a small **predict wrapper** that (a) encodes raw IDs with the same encoders, (b) forwards through the model under `no_grad()`, and (c) inverse-scales the output. For **cold-start** IDs, I return a sensible fallback (e.g., global mean) until retraining.

**Impact:** This module gave me a **trainable, extensible recommender** with clear learning curves (train/val RMSE), faster iteration on **embedding size**, **regularization**, and **learning rate**, and a clean **predict path** for the app. Compared to pure KNN, it improved **stability**, reduced manual feature engineering, and set the stage for richer architectures (e.g., concatenating embeddings + MLP).

**Summary:** Module 3 modernized the recommender core—moving from memory-based heuristics to a compact **embedding model** that’s easier to tune, deploy, and extend, with reproducible evaluation and a robust inference path.

**Keywords/Skills:** PyTorch, `nn.Embedding`, Adam, MSE/RMSE, train/val/test split, LabelEncoder, MinMaxScaler, evaluation helper, CPU-only training, cold-start handling, reproducibility, inference wrapper.

Here’s how I built the **Evaluation & Embedding Comparison** module, told as the story of how the recommender grew up from “works” to “works and proves it.”

I started with a working pipeline—BoW features flowing into KNN, NMF, classifier/regressor heads, and an ANN. It produced sensible lists, but I couldn’t answer the question stakeholders always ask: *“How good are these recommendations—and are BERT embeddings actually better than BoW?”* That gap made this module necessary: a rigorous, in-app evaluation layer that lets me flip between **BoW** and **BERT** embeddings and see objective metrics instantly.

Implementation came in small, focused steps. First, I generated **semantic course embeddings** with Sentence-BERT (`all-MiniLM-L6-v2`) and saved them as `course_bert_emb.csv`. Next, I built **user embeddings** by aggregating the BERT vectors of courses each user rated, using a **weighted mean** (weights = rating) so 5-star courses influence the profile more than 3-stars. In Streamlit, I added a clean **radio toggle** (`BoW` ↔ `BERT`) that rewires `user_emb_path`/`item_emb_path` dynamically for **ANN / RegEmbd / ClassEmbd**. Then I created `evaluate.py`, implementing **Precision\@K, Recall\@K, Intra-list Diversity, Inter-list Diversity, Novelty,** and **Catalog Coverage**. A thin wrapper in `backend.py` orchestrates predictions and metric computation for a given user and K, so the UI shows results right under the recommendation table.

It wasn’t all smooth. I chased down classic bugs: keyword-argument mismatches to `predict_*`, a `KeyError: 'item'` because one model returned a **Series** instead of a DataFrame, and an annoying `NaN` cascade traced to empty/ill-shaped prediction lists and a capitalized `'User'` column. I normalized outputs (every `predict_*` now returns `pd.DataFrame({'item': …})`), fixed the column naming to `user/item/rating`, and added defensive checks so metrics never silently fail.

The impact was immediate. With a single click, I can **quantitatively** compare BoW vs BERT for any model and user. The module exposes trade-offs (e.g., BERT improving **recall** and **novelty**, BoW sometimes yielding higher **intra-list diversity**) and highlights coverage gaps or popularity bias. For demos, it turns a black-box list into an **auditable, explainable** result—hugely improving trust and iteration speed.

**Tech & keywords:** Streamlit, Sentence-BERT, cosine similarity, weighted aggregation, Precision\@K/Recall\@K, diversity metrics, novelty, catalog coverage, pandas, scikit-learn, PyTorch/Surprise (KNN/NMF), modular dispatch in `backend.py`.

**In one line:** this module transformed the app from “it recommends” to “it recommends, explains, and proves it,” enabling fair BoW vs BERT comparisons and data-driven model choices.

Here’s the story of **Upgrade-1: the evaluation + ANN recommender refactor**—the piece that quietly turned a flaky prototype into a measurable, resume-worthy system.

I started with a deceptively simple problem: our Streamlit demo showed recommendations, but the **metrics were NaN** and **catalog coverage was 0%**. That meant I couldn’t trust any “improvement,” because there was no ground truth to measure against. Digging in, I found three culprits: (1) **ID drift**—course IDs in training (e.g., `RP0103`) didn’t match the “actuals” in ratings (e.g., `RP0103EN` / `…v1`); (2) **evaluation leakage**—we masked “seen” items using the entire ratings file, then tried to compare predictions against the same file, making overlap impossible; and (3) brittle plumbing—`Series` vs `list` types, `mean([])` warnings, and `Precision@k` computed on `set(p)` instead of `set(p[:k])`.

I rebuilt the pipeline step by step. Using **pandas** and **NumPy**, I introduced **canonical item IDs** end-to-end and persisted the mapping in the **PyTorch** artifact (`ann_model.pkl`). Next, I added a **user-level train/test split** (80/20) and wrote both to disk (`train_ratings.csv`, `test_ratings.csv`) to make experiments **reproducible**. The **ANN (PyTorch)** itself uses learned **user/item embeddings**, biases, **Adam** optimizer, and **MSE loss**; at inference, I score **all items** for a user, then **mask only train-seen items** so hits against the held-out test set are actually possible. In `evaluate.py`, I normalized output types (DataFrame/Series/list), fixed **Precision\@k/Recall\@k** to slice `p[:k]`, added guardrails to diversity functions to avoid `mean([])`, and applied **+1 smoothing** for novelty (`log2(popularity + 1)`). I also added lightweight debug logs to trace `recs`, `actuals`, and `Overlap`.

Challenges were half detective work, half engineering. The ID mismatch was subtle—the model looked fine but overlap was always zero. The fix was to **standardize IDs at data load, training, and prediction**. Another trap: passing the wrong frame into metrics (`ratings_df` vs `test_df`) or computing novelty with zero counts. I hardened everything with **type checks**, **early continues**, and **clear error messages**.

Impact: metrics are now **stable and trustworthy** (no NaNs), **coverage** is meaningful, **diversity/novelty** compute reliably, and the app shows an expandable **offline evaluation** right beside recommendations. More importantly, the refactor unlocked iteration: I can now tune epochs, embedding size, or even swap models and watch **Precision/Recall** respond—because the evaluation is correct.

**Keywords/skills:** PyTorch, embeddings, matrix factorization, ANN, pandas, NumPy, scikit-learn metrics, Streamlit, user-based train/test split, ID normalization, logging/debugging, MLOps hygiene, reproducibility, evaluation design (Precision\@k, Recall\@k, coverage, diversity, novelty), error handling, data plumbing.

**Summary:** this module transformed the project from a “looks okay” demo into a **measurable recommender** with clean data flow, robust metrics, and reproducible experiments—setting the stage for genuine model improvements.

# Qualitative Evaluation & Explainability Module (BoW vs BERT · t-SNE/UMAP · Side-by-Side Recs · Feedback)

When my offline metrics (Precision\@k, Recall\@k) kept reading near-zero—despite meaningful upgrades—I realized the project needed a different lens. Sparse ratings and hidden positives were poisoning the evaluation. So I built a **qualitative evaluation and explainability module** that lets me *see* and *compare* how models behave, rather than trusting brittle numbers.

I started by wiring a clean **dispatch architecture** in `backend.py` so every model (KNN, NMF, ClassEmbd, RegEmbd, ANN) could be called through a single `predict()` path. I added a lightweight `predict_with_embed_type(model, user, embed_type)` wrapper to switch **embedding sources** (BoW/TF-IDF vs **BERT**) without touching UI code. On the Streamlit side, I created a **side-by-side comparison** panel: choose User → pick Model A (BoW) and Model B (BERT) → render both Top-N lists together. To make decisions interpretable, I overlaid **user-specific** recommendations on the global item manifold using **t-SNE** (deterministic settings: `init="pca"`, fixed `random_state`) and added **UMAP** (via `umap-learn`) as a faster, structure-preserving alternative. The plots show all items faintly in gray and highlight only the user’s recommended items in color, so clusters and coverage are obvious at a glance.

This wasn’t plug-and-play. I hit a nasty **KeyError** when BERT CSVs didn’t match the feature schema (`UFeature*`, `CFeature*`); I fixed it by normalizing column names and making the predictors robust to naming. A **TypeError** surfaced from passing the wrong kwargs to `predict_nmf()`; I solved it by **model-aware parameter filtering**. I also discovered identical BoW/BERT results—root cause: some predictors ignored dynamic paths; I audited file loads to ensure **no hardcoded embedding paths**. Finally, I stabilized t-SNE variability and normalized axes so BoW vs BERT plots are truly comparable.

Impact: recruiters and stakeholders can now **visually verify diversity, novelty, and topical fit** per user; I can justify model/embedding choices with concrete, human-readable evidence; and a **feedback logger** (CSV) closes the loop for future learning. Technically, this module showcases **Streamlit**, **pandas**, **NumPy**, **scikit-learn (t-SNE)**, **UMAP**, **matplotlib**, **cosine similarity**, **BERT/TF-IDF embeddings**, and a clean **predict/train dispatch** pattern.

**Summary:** this feature turned a black-box recommender into an **interactive, explainable system**—making comparisons credible, debugging faster, and the overall user experience far more persuasive.
