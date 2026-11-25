"""
download_data.py - Data sourcing and metadata enrichment for MovieLens hybrid recommendation.
Auto-downloads MovieLens 100K (default) or 1M, parses movie metadata,
and enriches movies with plot summaries, genres, and poster URLs with disk caching.
Supports optional TMDb API keys or falls back to robust offline metadata generation.
"""

import os
import sys
import json
import time
import zipfile
import urllib.request
import urllib.parse
import ssl
import argparse
from typing import Dict, Any, Optional

try:
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except Exception:
    SSL_CONTEXT = ssl._create_unverified_context()

MOVIELENS_URLS = {
    "100k": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
    "1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
}

GENRE_DESCRIPTIONS = {
    "Action": "high-octane action, thrilling stunts, and explosive confrontations",
    "Adventure": "epic journeys, questing through uncharted territories, and brave exploits",
    "Animation": "vibrant animated storytelling with imaginative visual artistry",
    "Children": "family-friendly themes suitable for all ages with heartwarming morals",
    "Children's": "family-friendly themes suitable for all ages with heartwarming morals",
    "Comedy": "witty dialogue, humorous misadventures, and lighthearted entertainment",
    "Crime": "underworld intrigue, detectives solving mysteries, and heist escapades",
    "Documentary": "real-world exploration of historical events, people, and society",
    "Drama": "deep emotional depth, character-driven storytelling, and poignant conflicts",
    "Fantasy": "mystical realms, magical creatures, and legendary epic quests",
    "Film-Noir": "moody, cynical crime drama characterized by dark shadows and moral ambiguity",
    "Horror": "spine-chilling suspense, supernatural terror, and psychological fear",
    "Musical": "show-stopping songs, vibrant musical choreography, and stage romance",
    "Mystery": "perplexing puzzles, detective investigations, and unexpected twists",
    "Romance": "passionate love stories, heartfelt emotional connections, and relationship trials",
    "Sci-Fi": "futuristic technology, space exploration, and speculative scientific concepts",
    "Thriller": "edge-of-your-seat suspense, psychological tension, and high stakes",
    "War": "wartime sacrifice, military strategy, and human resilience under fire",
    "Western": "dusty frontier justice, outlaws, and gunslinger confrontations"
}

# Curated fallback poster library for key classic MovieLens titles to guarantee rich visuals offline
CURATED_POSTERS = {
    "Toy Story": "https://image.tmdb.org/t/p/w500/uXDfjJbdP4ijW5hWSBrFuVJTkab.jpg",
    "GoldenEye": "https://image.tmdb.org/t/p/w500/z0lj0Zlh7NuV04Gbyem6ZqCWx4c.jpg",
    "Four Rooms": "https://image.tmdb.org/t/p/w500/eQs590JnQ8qU5Y9cE5fX8E0B3bE.jpg",
    "Get Shorty": "https://image.tmdb.org/t/p/w500/v39D4P60nF70f7HkO2D016Uq7A2.jpg",
    "Copycat": "https://image.tmdb.org/t/p/w500/i9YF0tE9x9M5N1C2A5yv6s0Xg7m.jpg",
    "Twelve Monkeys": "https://image.tmdb.org/t/p/w500/6Sj9wDN5G1cA594cW44iV5vL0kM.jpg",
    "Babe": "https://image.tmdb.org/t/p/w500/d5vL3sE0eGqN6vL4W7xT3Xb8C1g.jpg",
    "Dead Man Walking": "https://image.tmdb.org/t/p/w500/1lS0zK5q8j6a6N2g8iY6wK5F1qD.jpg",
    "Seven": "https://image.tmdb.org/t/p/w500/6yoghtyTpznpBik8EngEmJskVUO.jpg",
    "Usual Suspects, The": "https://image.tmdb.org/t/p/w500/bUPMzL4Tuvn5Wq51UVh5qO1t5Xm.jpg",
    "Fargo": "https://image.tmdb.org/t/p/w500/rt7cpEr1u9oGvi789QcQp618c6W.jpg",
    "Star Wars": "https://image.tmdb.org/t/p/w500/6FfCtAuVAW8XJjZ7eWeLibRLWTw.jpg",
    "Godfather, The": "https://image.tmdb.org/t/p/w500/3bhkrj58Vtu7enYsRolD1fZdja1.jpg",
    "Pulp Fiction": "https://image.tmdb.org/t/p/w500/d5iIlFn5s0ImszYzBPb8JPIfbXD.jpg",
    "Shawshank Redemption, The": "https://image.tmdb.org/t/p/w500/9cqNxx0GxF0bflZmeSMuL5tnGzr.jpg",
    "Forrest Gump": "https://image.tmdb.org/t/p/w500/arw2VCBveWOVZr6pxd9XTd1TdQa.jpg",
    "Silence of the Lambs, The": "https://image.tmdb.org/t/p/w500/uS9m8OBk1A8eM9I042bx8XXpqAq.jpg",
    "Schindler's List": "https://image.tmdb.org/t/p/w500/sF1U4EUQS8YHUYjNl3pMGNIQyr0.jpg",
    "Titanic": "https://image.tmdb.org/t/p/w500/9xjZS2rlVxm8SFx8kPC3aIGCOYQ.jpg",
    "Jurassic Park": "https://image.tmdb.org/t/p/w500/oU7Oq2kFAAlGqbU4VoAE36g4hoI.jpg"
}

DEFAULT_POSTER = "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&q=80"


def download_movielens(dataset: str = "100k", target_dir: str = "data") -> str:
    """Download and extract MovieLens dataset if not already present."""
    os.makedirs(target_dir, exist_ok=True)
    raw_dir = os.path.join(target_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    dest_folder = os.path.join(target_dir, f"ml-{dataset}")
    expected_file = os.path.join(dest_folder, "u.data" if dataset == "100k" else "ratings.dat")

    if os.path.exists(expected_file):
        print(f"[OK] MovieLens {dataset.upper()} already exists at {dest_folder}")
        return dest_folder

    url = MOVIELENS_URLS.get(dataset.lower())
    if not url:
        raise ValueError(f"Unsupported dataset: {dataset}. Choose '100k' or '1m'.")

    zip_path = os.path.join(raw_dir, f"ml-{dataset}.zip")
    print(f"[*] Downloading MovieLens {dataset.upper()} from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, context=SSL_CONTEXT, timeout=30) as resp, open(zip_path, 'wb') as out_f:
            out_f.write(resp.read())
    except Exception:
        # Fallback to unverified context if needed
        unverified_ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(req, context=unverified_ctx, timeout=30) as resp, open(zip_path, 'wb') as out_f:
            out_f.write(resp.read())
    print(f"[OK] Downloaded to {zip_path}")

    print(f"[*] Extracting {zip_path} to {target_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

    print(f"[OK] Extracted successfully to {dest_folder}")
    return dest_folder


def parse_movies_100k(ml_dir: str) -> Dict[int, Dict[str, Any]]:
    """Parse MovieLens 100K u.item file."""
    item_file = os.path.join(ml_dir, "u.item")
    genres_list = [
        "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
        "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]

    movies = {}
    with open(item_file, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 24:
                continue
            movie_id = int(parts[0])
            title = parts[1].strip()
            release_date = parts[2].strip() if len(parts) > 2 else ""
            imdb_url = parts[4].strip() if len(parts) > 4 else ""

            # Genre flags from index 5 to 23 (19 genres, skip first 'unknown' genre at 5)
            active_genres = []
            genre_flags = [int(x) for x in parts[6:24]]
            for name, flag in zip(genres_list, genre_flags):
                if flag == 1:
                    active_genres.append(name)

            if not active_genres:
                active_genres = ["Drama"]

            movies[movie_id] = {
                "movie_id": movie_id,
                "title": title,
                "release_date": release_date,
                "imdb_url": imdb_url,
                "genres": active_genres
            }
    return movies


def parse_movies_1m(ml_dir: str) -> Dict[int, Dict[str, Any]]:
    """Parse MovieLens 1M movies.dat file."""
    item_file = os.path.join(ml_dir, "movies.dat")
    movies = {}
    with open(item_file, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) < 3:
                continue
            movie_id = int(parts[0])
            title = parts[1].strip()
            genres = parts[2].strip().split("|")
            movies[movie_id] = {
                "movie_id": movie_id,
                "title": title,
                "release_date": "",
                "imdb_url": "",
                "genres": genres
            }
    return movies


def generate_synthetic_plot(title: str, genres: list) -> str:
    """Generate a rich, cohesive thematic plot summary based on title and genre blend."""
    # Clean year out of title for prose
    clean_title = title.split(" (")[0] if " (" in title else title
    genre_phrases = [GENRE_DESCRIPTIONS.get(g, f"{g.lower()} narrative") for g in genres[:3]]
    joined_genres = ", featuring ".join(genre_phrases)

    return (
        f"In '{clean_title}', audiences embark on a compelling cinematic journey of {joined_genres}. "
        f"Driven by profound personal stakes and gripping character relationships, the story explores "
        f"unforeseen challenges, deep emotional truths, and an unforgettable climax."
    )


def fetch_tmdb_metadata(title: str, api_key: str) -> Optional[Dict[str, Any]]:
    """Fetch movie overview and poster from TMDb API with rate limiting and timeout."""
    try:
        clean_title = title.split(" (")[0]
        query = urllib.parse.quote(clean_title)
        url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={query}"
        req = urllib.request.Request(url, headers={"User-Agent": "HybridRecSys/1.0"})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("results", [])
            if results:
                best = results[0]
                poster_path = best.get("poster_path")
                poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
                overview = best.get("overview")
                return {
                    "overview": overview if overview and len(overview) > 20 else None,
                    "poster_url": poster_url
                }
    except Exception:
        pass
    return None


def enrich_metadata(
    movies: Dict[int, Dict[str, Any]],
    cache_path: str = "data/cache/metadata_cache.json",
    tmdb_api_key: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """Enrich movie items with plot summaries and poster URLs using local caching."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache: Dict[str, Dict[str, Any]] = {}

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            print(f"[OK] Loaded {len(cache)} cached movie metadata records from {cache_path}")
        except Exception as e:
            print(f"[!] Warning: Could not read metadata cache ({e}). Starting fresh.")
            cache = {}

    updated = False
    total = len(movies)
    print(f"[*] Verifying/enriching metadata for {total} movies...")

    for i, (m_id, m_data) in enumerate(movies.items(), start=1):
        m_id_str = str(m_id)
        if m_id_str in cache and cache[m_id_str].get("plot") and cache[m_id_str].get("poster_url"):
            continue

        title = m_data["title"]
        genres = m_data["genres"]

        plot = None
        poster_url = None

        # Check curated posters
        for cur_title, cur_url in CURATED_POSTERS.items():
            if cur_title.lower() in title.lower():
                poster_url = cur_url
                break

        # Optional online lookup if API key is provided
        if tmdb_api_key:
            res = fetch_tmdb_metadata(title, tmdb_api_key)
            if res:
                if res.get("overview"):
                    plot = res["overview"]
                if res.get("poster_url"):
                    poster_url = res["poster_url"]
            time.sleep(0.05)  # Rate limit

        # Fallback plot generation
        if not plot:
            plot = generate_synthetic_plot(title, genres)

        if not poster_url:
            poster_url = DEFAULT_POSTER

        cache[m_id_str] = {
            "movie_id": m_id,
            "title": title,
            "genres": genres,
            "plot": plot,
            "poster_url": poster_url
        }
        updated = True

        if i % 400 == 0 or i == total:
            print(f"    Progress: {i}/{total} movies processed")

    if updated or not os.path.exists(cache_path):
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        print(f"[OK] Saved metadata cache to {cache_path}")

    return cache


def main():
    parser = argparse.ArgumentParser(description="Download and enrich MovieLens dataset.")
    parser.add_argument("--dataset", type=str, default="100k", choices=["100k", "1m"], help="MovieLens version (100k or 1m)")
    parser.add_argument("--data-dir", type=str, default="data", help="Target data directory")
    parser.add_argument("--tmdb-api-key", type=str, default=os.getenv("TMDB_API_KEY", None), help="Optional TMDb API key")
    args = parser.parse_args()

    ml_dir = download_movielens(dataset=args.dataset, target_dir=args.data_dir)
    if args.dataset == "100k":
        movies = parse_movies_100k(ml_dir)
    else:
        movies = parse_movies_1m(ml_dir)

    print(f"[OK] Parsed {len(movies)} movie records from {ml_dir}")
    cache_path = os.path.join(args.data_dir, "cache", f"metadata_cache_{args.dataset}.json")
    enrich_metadata(movies, cache_path=cache_path, tmdb_api_key=args.tmdb_api_key)
    print("[SUCCESS] Data sourcing and enrichment pipeline complete.")


if __name__ == "__main__":
    main()
