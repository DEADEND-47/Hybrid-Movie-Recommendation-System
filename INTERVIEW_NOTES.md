# CineMatch AI: Technical Interview Preparation Sheet

This cheat sheet provides concise, defensible answers (3–5 sentences each) to the core system design, algorithmic, and evaluation questions most likely to arise in AI Engineering internship interviews.

---

### Q1: Why did you build a hybrid system instead of picking a single strong model like NCF or SVD?
> Single-paradigm models suffer from complementary failure modes: collaborative models (NCF, SVD) excel at mining subtle user behavior patterns but collapse when users or items have zero ratings, while content-based models generalize to new metadata but ignore community preference signals. A weighted hybrid combines the non-linear interaction capacity of NCF, the global bias regularization of SVD, and the semantic generalization of TF-IDF. Empirically, our multi-seed evaluation shows that hybrid blending outperforms standalone SVD ($0.0597 \pm 0.0014$ NDCG@10) and standalone Content ($0.0176 \pm 0.0011$) on full-catalog ranking ($0.0661 \pm 0.0018$ NDCG@10) while maintaining strong rating prediction ($0.9177 \pm 0.0075$ RMSE).

---

### Q2: How did you validate that the hybrid actually helps, rather than just asserting it?
> We validated the hybrid along three dimensions: trivial baseline comparison, full-catalog ranking evaluation, and multi-seed stability with leak-free holdout splits. First, all collaborative and hybrid models decisively beat Global-Mean ($1.1292$ RMSE) and User-Mean ($1.0495$ RMSE) baselines, achieving $\approx 0.92$ RMSE. Second, instead of evaluating ranking only on a user's test items, we ranked across the entire unrated catalog ($\approx 1,591$ candidates per user), showing that the default hybrid achieves $4.78\%$ Precision@10 ($11.1\times$ random chance at $0.43\%$). Third, across three independent training/evaluation splits (`42`, `123`, `999`), the hybrid consistently beats standalone SVD in NDCG@10 ($0.0661 \pm 0.0018$ vs. $0.0597 \pm 0.0014$) by over $3.5\times$ standard deviation.

---

### Q3: How do you handle cold start, and what does your multi-seed alpha sweep show?
> When an unrated user enters the system, `HybridRecommender` reads `cold_start_alpha` from `config.yaml` and scores items via $\alpha \cdot \text{CosineSim}_{\text{content}} + (1 - \alpha) \cdot \text{ItemBias}_{\text{svd}}$. In a 3-seed cold-start sweep across 50 randomly sampled users per seed ($1,679$ catalog candidates), pure content ($\alpha = 1.0$) underperforms even popularity ($0.0060 \pm 0.0048$ vs. $0.0274 \pm 0.0056$ NDCG@10) because 3 seed ratings produce an overly narrow lexical filter. Blending across the plateau $\alpha \in [0.4, 0.7]$ decisively beats pure content ($>3\sigma$) and raises mean NDCG@10 ($0.0323\text{–}0.0364$) and Recall@10 ($0.0380\text{–}0.0437$) above pure popularity within overlapping std bands; we configure $\alpha = 0.5$ in `config.yaml` as a reasonable, low-variance default (`P@10 = 0.0253 ± 0.0009`) rather than claiming a sharp unique optimum.

---

### Q4: Why did you choose 32-dimensional embeddings, and why this specific evaluation protocol?
> We started with $K=32$ for GMF and $K=32$ for MLP (concatenated into 64-dim) because MovieLens 100K has only 943 users and 1,682 movies; larger embedding dimensions (e.g., 128+) create excessive capacity that rapidly overfits within 3 to 4 epochs on this dataset scale. Furthermore, $K=32$ easily fits inside our strict $<4.0\text{ GB}$ VRAM constraint on the RTX 3050, consuming only $20.6\text{ MB}$ of peak VRAM during AMP training. For evaluation, ranking against all unrated catalog items ($|C_u| \approx 1,591$) rather than a restricted 10-item test subset ensures the candidate pool matches real-world retrieval conditions where models must filter a large catalog of distractors.

---

### Q5: What would you change to make this system handle $10\times$ to $100\times$ the data (MovieLens 1M / 20M)?
> Scaling from 100K to 20M interactions requires transitioning from single-stage scoring to a two-stage **Retrieval and Re-Ranking** pipeline. First, the $O(M^2)$ pairwise cosine similarity matrix must be replaced with Approximate Nearest Neighbor (ANN) search using FAISS, HNSW, or ScaNN to reduce vector lookups from linear scan to sub-linear logarithmic time. Second, NCF should be trained using negative sampling under implicit feedback (e.g., binary cross-entropy or BPR loss) with PyTorch DistributedDataParallel (DDP) across mini-batches. Third, the real-time inference layer should precompute item embeddings offline, store them in a feature store or vector database, and perform dot-product retrieval at the service layer rather than re-computing similarities on the fly.

---

### Q6: What is the single biggest limitation of this system as built today?
> The biggest limitation is that the system operates exclusively on **explicit ratings** $[1, 5]$ optimized for Mean Squared Error, whereas real-world production recommender systems primarily operate on **implicit feedback** (clicks, watch time, impressions, add-to-watchlist). Explicit ratings suffer from severe selection bias—users predominantly rate movies they already expected to like—whereas implicit signals provide dense behavioral traces across the entire catalog. Transitioning the neural loss to a pairwise ranking objective (like Bayesian Personalized Ranking or InfoNCE) and ingesting session-based sequential actions would significantly improve real-time engagement compared to static rating regression.

---

### Q7: Walk me through a subtle bug you caught and fixed during evaluation.
> We caught two major evaluation bugs: first, a **train-test contamination bug** where seed-42 checkpoints were evaluated against seeds 123 and 999 without retraining (deflating Hybrid RMSE from `0.9177` down to an artificial `0.8611`), fixed via `train_multiseed.py`; second, `NCFRecommender.load()` omitted `model.eval()`, leaving `Dropout(0.2)` active during test-pair RMSE computation (`0.9279` vs. true `0.9166` RMSE, a `-0.0113` delta). We also confirmed `app.py` and `api.py` remain pinned strictly to the canonical seed-42 artifacts.

---

### Q8: Are your hybrid weight choices (0.45/0.35/0.20) empirically proven optimal in your ablation study?
> No — our multi-seed ablation honestly shows that weight choice has limited effect at the MovieLens 100K scale. Across 3 seeds, `collaborative_only` ($0.50/0.50/0.00$) achieves slightly better RMSE ($0.9129 \pm 0.0064$) and `equal_blend` ($0.333/0.333/0.334$) achieves slightly higher NDCG@10 ($0.0673 \pm 0.0022$), and both fall strictly inside the standard deviation noise band of `tuned_hybrid` (RMSE $0.9177 \pm 0.0075$, NDCG $0.0661 \pm 0.0018$). Rather than overselling a tuned optimum, we defend `0.45/0.35/0.20` as a reasonable engineering default—keeping 80% weight on collaborative filtering while retaining 20% content weight for diversity—whereas the statistically significant gain is simply using a collaborative-content blend over standalone SVD ($0.0597 \pm 0.0014$ NDCG@10).
