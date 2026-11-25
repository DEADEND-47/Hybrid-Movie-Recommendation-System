"""
app.py - Interactive Streamlit Web Application for Hybrid Movie Recommendation System.
Features:
  - Interactive movie rating widget (3-5 movies rated by user)
  - Real-time local inference via Weighted Hybrid Recommender (NCF + SVD + Content)
  - Top-10 personalized recommendations with posters, metadata, and score breakdown
  - Dynamic weight tuner in sidebar
  - Cold-Start Demonstration tab for brand-new movies or users with zero historical interactions
"""

import os
import yaml
import pickle
import numpy as np
import pandas as pd
import streamlit as st

import torch
from models.content import ContentRecommender
from models.svd import SVDRecommender
from models.ncf import NCFRecommender
from models.hybrid import HybridRecommender


def load_app_config(config_path: str = "config.yaml") -> dict:
    """Load centralized settings from config.yaml."""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

# Set Page Config
st.set_page_config(
    page_title="CineMatch AI - Hybrid Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #E50914, #B81D24, #221F1F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #718096;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.4rem;
        background-color: #2D3748;
        color: #ED8936;
    }
    .movie-card {
        background-color: #1A202C;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #2D3748;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading trained recommendation models...")
def load_system_artifacts(artifacts_dir: str = "artifacts", config_path: str = "config.yaml"):
    """Cache-load models, mappings, and config.yaml defaults for instant local inference."""
    cfg = load_app_config(config_path)
    hybrid_cfg = cfg.get("models", {}).get("hybrid", {})
    default_weights = hybrid_cfg.get("default_weights", {"ncf": 0.45, "svd": 0.35, "content": 0.20})
    cold_start_alpha = float(hybrid_cfg.get("cold_start_alpha", 0.5))

    mappings_path = os.path.join(artifacts_dir, "user_item_maps.pkl")
    content_path = os.path.join(artifacts_dir, "content_engine.pkl")
    svd_path = os.path.join(artifacts_dir, "svd_model.pkl")
    ncf_path = os.path.join(artifacts_dir, "ncf_model.pt")

    if not all(os.path.exists(p) for p in [mappings_path, content_path, svd_path, ncf_path]):
        st.error("Model artifacts not found! Please run 'python train.py' first.")
        st.stop()

    with open(mappings_path, "rb") as f:
        maps = pickle.load(f)

    content_model = ContentRecommender.load(content_path)
    svd_model = SVDRecommender.load(svd_path)
    ncf_model = NCFRecommender.load(ncf_path)
    hybrid_model = HybridRecommender(
        content_model,
        svd_model,
        ncf_model,
        default_weights=default_weights,
        cold_start_alpha=cold_start_alpha,
        config_path=config_path
    )

    return maps, content_model, svd_model, ncf_model, hybrid_model, cfg


def main():
    maps, content_model, svd_model, ncf_model, hybrid_model, cfg = load_system_artifacts()
    movies_df = content_model.movies_df

    st.markdown('<div class="main-header">🎬 CineMatch AI: Hybrid Movie Recommender</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">B.Tech CSE (AI) Capstone Project | PyTorch NCF (NeuMF) + Classical SVD + Content-Based TF-IDF Fusion</div>', unsafe_allow_html=True)

    # Sidebar: System Controls & Weights
    st.sidebar.header("⚙️ Model Weights & Fusion")
    st.sidebar.markdown("Customize the fusion contribution of each recommendation engine:")

    cfg_w = hybrid_model.default_weights
    weight_preset = st.sidebar.selectbox(
        "Weight Strategy Preset",
        ["Default Balanced (45/35/20)", "Neural Dominant (65/25/10)", "Matrix Factorization Dominant (20/60/20)", "Content-Heavy (15/15/70)", "Custom"]
    )

    if weight_preset == "Default Balanced (45/35/20)":
        w_ncf, w_svd, w_content = cfg_w["ncf"], cfg_w["svd"], cfg_w["content"]
    elif weight_preset == "Neural Dominant (65/25/10)":
        w_ncf, w_svd, w_content = 0.65, 0.25, 0.10
    elif weight_preset == "Matrix Factorization Dominant (20/60/20)":
        w_ncf, w_svd, w_content = 0.20, 0.60, 0.20
    elif weight_preset == "Content-Heavy (15/15/70)":
        w_ncf, w_svd, w_content = 0.15, 0.15, 0.70
    else:
        w_ncf = st.sidebar.slider("NCF (NeuMF) Weight", 0.0, 1.0, float(cfg_w["ncf"]), 0.05)
        w_svd = st.sidebar.slider("SVD Weight", 0.0, 1.0, float(cfg_w["svd"]), 0.05)
        w_content = st.sidebar.slider("Content Weight", 0.0, 1.0, float(cfg_w["content"]), 0.05)

    weights = {"ncf": w_ncf, "svd": w_svd, "content": w_content}
    cold_start_alpha = st.sidebar.slider(
        "Cold-Start Alpha (Content vs Item Bias)",
        0.0, 1.0, float(hybrid_model.cold_start_alpha), 0.1,
        help="alpha * ContentSim + (1 - alpha) * SVD Item Bias. Default loaded from config.yaml."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Hardware Acceleration:** `{'NVIDIA RTX 3050 (AMP)' if torch.cuda.is_available() else 'CPU'}`")
    st.sidebar.markdown(f"**Indexed Movies:** `{len(movies_df):,}`")
    st.sidebar.markdown(f"**Training Matrix Users:** `{maps['num_users']:,}`")
    st.sidebar.markdown(f"**Config Cold-Start Alpha:** `{hybrid_model.cold_start_alpha}`")

    # App Tabs
    tab_rec, tab_cold_start, tab_arch = st.tabs([
        "🎯 Personalized Recommendations",
        "❄️ Cold-Start Demonstration",
        "📊 Architecture & Performance"
    ])

    # TAB 1: Personalized Recommendations
    with tab_rec:
        st.subheader("1. Rate 3 to 5 Movies to Seed Your Profile")
        st.write("Select familiar movies from the dropdown and assign your ratings (1 to 5 stars).")

        # Select initial sample movies
        popular_titles = [
            "Toy Story (1995)", "Star Wars (1977)", "Fargo (1996)",
            "Godfather, The (1972)", "Pulp Fiction (1994)", "Twelve Monkeys (1995)",
            "Silence of the Lambs, The (1991)", "Jurassic Park (1993)", "Titanic (1997)"
        ]
        available_titles = movies_df["title"].tolist()
        default_selections = [t for t in popular_titles if t in available_titles][:3]

        selected_movies = st.multiselect(
            "Search and select movies to rate:",
            options=available_titles,
            default=default_selections,
            max_selections=8
        )

        user_ratings = {}
        if selected_movies:
            cols = st.columns(min(len(selected_movies), 4))
            for idx, title in enumerate(selected_movies):
                col = cols[idx % len(cols)]
                with col:
                    row = movies_df[movies_df["title"] == title].iloc[0]
                    item_idx = int(row["item_idx"])
                    poster = row["poster_url"] if row["poster_url"] else "https://via.placeholder.com/300x450?text=No+Poster"
                    st.image(poster, use_container_width=True)
                    st.caption(f"**{title}**")
                    rating_val = st.slider(f"Rating", 1.0, 5.0, 4.5, 0.5, key=f"rate_{item_idx}")
                    user_ratings[item_idx] = float(rating_val)

        st.markdown("---")
        col_btn, col_k = st.columns([2, 1])
        with col_k:
            top_k = st.slider("Number of recommendations", 5, 20, 10)
        with col_btn:
            get_rec = st.button("🚀 Generate Hybrid Recommendations", type="primary", use_container_width=True)

        if get_rec or len(user_ratings) > 0:
            st.subheader(f"Top-{top_k} Recommendations for You")
            with st.spinner("Executing hybrid fusion inference across NCF, SVD, and Content models..."):
                recs = hybrid_model.recommend(
                    u_idx=None,  # Interactive user is treated as dynamic cold-start profile
                    user_ratings=user_ratings,
                    top_k=top_k,
                    weights=weights,
                    exclude_rated=True,
                    cold_start_alpha=cold_start_alpha
                )

            # Display cards in grid
            rec_cols = st.columns(5)
            for i, rec in enumerate(recs):
                col = rec_cols[i % 5]
                with col:
                    poster = rec["poster_url"] if rec["poster_url"] else "https://via.placeholder.com/300x450?text=No+Poster"
                    st.image(poster, use_container_width=True)
                    st.markdown(f"**#{i+1} {rec['title']}**")
                    st.caption(f"🏷️ {rec['genres']}")
                    st.markdown(f"⭐ **Pred: {rec['predicted_rating']:.2f} / 5.0**")
                    st.markdown(f"<span class='metric-badge'>Score: {rec['hybrid_score']:.3f}</span>", unsafe_allow_html=True)
                    with st.expander("Score Breakdown"):
                        st.write(f"• Content Sim: `{rec['component_scores']['content_score']:.3f}`")
                        st.write(f"• SVD Rating: `{rec['component_scores']['svd_rating']:.2f}`")
                        st.write(f"• NCF Rating: `{rec['component_scores']['ncf_rating']:.2f}`")
                        st.write(f"• Mode: `{rec['routing_mode']}`")
                        st.write(f"• Synopsis: *{rec['plot']}*")

    # TAB 2: Cold-Start Demonstration
    with tab_cold_start:
        st.subheader("❄️ Cold-Start Mitigation Engine")
        st.write(
            "Collaborative filtering (NCF and SVD) fails when an item or user has **zero historical ratings** (the classic Cold-Start Problem). "
            "Our hybrid system dynamically routes unrated entities to the **TF-IDF Content Engine**, ensuring zero downtime or recommendation blackout."
        )

        st.markdown("#### Test a Brand-New Unrated Movie (Simulated Zero Interactions)")
        col_new1, col_new2 = st.columns([1, 1])

        with col_new1:
            test_title = st.text_input("New Movie Title", value="Chrono Nexus: AI Genesis (2026)")
            test_genres = st.multiselect(
                "Movie Genres",
                options=["Action", "Adventure", "Animation", "Comedy", "Crime", "Drama", "Fantasy", "Horror", "Mystery", "Sci-Fi", "Thriller"],
                default=["Sci-Fi", "Thriller", "Action"]
            )
            test_synopsis = st.text_area(
                "Plot Synopsis",
                value="In a dystopian metropolis governed by an enigmatic quantum artificial intelligence, a rogue neural engineer discovers a temporal loop threatening to erase human autonomy."
            )
            simulate_btn = st.button("🔍 Recommend Similar Movies for this Unrated Title", type="primary")

        with col_new2:
            st.info("💡 **How CineMatch AI Resolves Cold-Start:**\n\n"
                    "1. Since this movie has **0 ratings**, it has no embedding vector in NCF or latent vector in SVD.\n"
                    "2. The system vectorizes the new title, genres, and synopsis through the fitted TF-IDF matrix.\n"
                    "3. Cosine similarity calculates nearest semantic neighbors in real-time.\n"
                    "4. The hybrid router shifts weights to **100% Content**, guaranteeing high-quality recommendations immediately.")

        if simulate_btn:
            soup = f"{test_title} {' '.join(test_genres)} {test_synopsis}"
            new_vec = content_model.tfidf_vectorizer.transform([soup])
            # Cosine similarity against all existing movies
            from sklearn.metrics.pairwise import cosine_similarity
            sims = cosine_similarity(new_vec, content_model.tfidf_matrix).flatten()
            top_sim_idx = np.argsort(sims)[::-1][:6]

            st.markdown("---")
            st.subheader(f"Top Movies Semantically Aligned with '{test_title}'")
            cs_cols = st.columns(6)
            for idx, m_idx in enumerate(top_sim_idx):
                col = cs_cols[idx]
                row = movies_df.iloc[m_idx]
                with col:
                    poster = row["poster_url"] if row["poster_url"] else "https://via.placeholder.com/300x450?text=No+Poster"
                    st.image(poster, use_container_width=True)
                    st.markdown(f"**{row['title']}**")
                    st.caption(f"🏷️ {row['genres']}")
                    st.markdown(f"✨ **Similarity: {sims[m_idx]:.3f}**")
                    st.write(f"*{row['plot'][:110]}...*")

    # TAB 3: Architecture & Performance
    with tab_arch:
        st.subheader("System Architecture & Offline Benchmark")

        col_a1, col_a2 = st.columns([1, 1])
        with col_a1:
            st.markdown(r"""
            ### 🏗️ Model Architecture Overview
            - **Neural Collaborative Filtering (NeuMF):**
              - **GMF Branch:** 32-dim user & item embeddings with element-wise product.
              - **MLP Branch:** 32-dim user & item embeddings concatenated to 64-dim, passed through 3 dense layers (`128 -> 64 -> 32`) with ReLU, BatchNorm, and Dropout (0.2).
              - **Fusion Layer:** Concatenates GMF (32) + MLP (32) -> Linear projection to rating prediction.
              - **Training:** PyTorch with `torch.amp.autocast('cuda')` mixed precision on RTX 3050 (Peak VRAM < 1.5 GB).
            - **Classical SVD (Biased Matrix Factorization):**
              - Baseline regularized SGD learning user biases $b_u$, item biases $b_i$, and factor matrices $P \in \mathbb{R}^{U \times 32}, Q \in \mathbb{R}^{I \times 32}$.
            - **Content-Based Filtering:**
              - TF-IDF 5,000 unigram/bigram vocabulary across plot synopsis and genres with cosine similarity matrix.
            - **Weighted Hybrid Fusion:**
              - Dynamically balances CF and Content scores with cold-start automatic fallback.
            """)

        with col_a2:
            metrics_path = "outputs/metrics.json"
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    eval_metrics = json.load(f)
                st.markdown("### 📈 Test Set Offline Benchmark (10% Holdout)")
                metrics_df = pd.DataFrame(eval_metrics).T
                st.dataframe(metrics_df.style.highlight_min(subset=["RMSE", "MAE"], color="#2D4A3E")
                                             .highlight_max(subset=[c for c in metrics_df.columns if "P@" in c or "R@" in c or "NDCG" in c], color="#2D4A3E"),
                             use_container_width=True)

            comp_img = "outputs/model_comparison.png"
            if os.path.exists(comp_img):
                st.image(comp_img, caption="Offline Model Comparison: RMSE/MAE Error & Top-10 Ranking NDCG", use_container_width=True)


if __name__ == "__main__":
    main()
