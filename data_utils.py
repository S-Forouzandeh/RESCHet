"""
data_utils.py — Dataset loaders for RESCHet

Supported datasets
──────────────────
• MovieLens (ml-latest-small)   — demo / development
• Douban Movie / Book           — paper benchmark
• Yelp                          — paper benchmark

All loaders return (hin: HIN, ratings_df: pd.DataFrame).

Node-ID conventions
────────────────────
    users      → "u_{original_id}"
    movies/items → "m_{original_id}"
    directors/aux → "d_{original_id_or_name}"
"""

import os
import pandas as pd
from reschet import HIN


# ─────────────────────────────────────────────────────────────────────────────
# MovieLens (ml-latest-small / ml-1m)
# ─────────────────────────────────────────────────────────────────────────────

def load_movielens(
    data_dir: str = "data/ml-latest-small",
    max_ratings: int = 50_000,
    use_genre_as_director: bool = True,
) -> tuple[HIN, pd.DataFrame]:
    """
    Build a User–Movie–Director HIN from the MovieLens dataset.

    When *use_genre_as_director* is True (default), the primary genre of
    each movie is treated as the "director" node — this keeps the
    three-node-type HIN schema from the paper without requiring an
    external director database.

    If you have a ``directors.csv`` (columns: movieId, director) in
    *data_dir*, set *use_genre_as_director=False* and it will be used
    instead.

    Download
    ────────
        wget https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
        unzip ml-latest-small.zip -d data/
    """
    movies_path = os.path.join(data_dir, "movies.csv")
    ratings_path = os.path.join(data_dir, "ratings.csv")
    directors_path = os.path.join(data_dir, "directors.csv")

    if not os.path.exists(movies_path):
        raise FileNotFoundError(f"movies.csv not found in {data_dir}")
    if not os.path.exists(ratings_path):
        raise FileNotFoundError(f"ratings.csv not found in {data_dir}")

    movies_df = pd.read_csv(movies_path)
    ratings_df = pd.read_csv(ratings_path)

    # ── optional sub-sample ──────────────────────────────────────────────────
    if len(ratings_df) > max_ratings:
        ratings_df = ratings_df.sample(max_ratings, random_state=42).reset_index(drop=True)

    # ── build auxiliary (director / genre) mapping ───────────────────────────
    if not use_genre_as_director and os.path.exists(directors_path):
        dir_df = pd.read_csv(directors_path)          # movieId, director
        aux_map: dict[int, str] = {}
        for _, row in dir_df.iterrows():
            aux_map[int(row["movieId"])] = str(row["director"])
    else:
        # Use first genre as the third node type
        aux_map = {}
        for _, row in movies_df.iterrows():
            genres = str(row.get("genres", "(no genres listed)")).split("|")
            genre = genres[0] if genres[0] != "(no genres listed)" else "Unknown"
            aux_map[int(row["movieId"])] = genre

    # ── build HIN ────────────────────────────────────────────────────────────
    hin = HIN()
    rated_users = set(ratings_df["userId"].unique())
    rated_movies = set(ratings_df["movieId"].unique())

    for uid in rated_users:
        hin.add_node(f"u_{uid}", "user")

    for mid in rated_movies:
        hin.add_node(f"m_{mid}", "movie")
        aux_val = aux_map.get(mid, "Unknown")
        d_node = f"d_{aux_val}"
        if d_node not in hin.node_type:
            hin.add_node(d_node, "director")

    # user–rates–movie
    for row in ratings_df.itertuples(index=False):
        uid = f"u_{row.userId}"
        mid = f"m_{row.movieId}"
        if uid in hin.node_type and mid in hin.node_type:
            hin.add_edge(uid, mid, "rates")

    # movie–has_director–director
    for mid in rated_movies:
        m_node = f"m_{mid}"
        d_node = f"d_{aux_map.get(mid, 'Unknown')}"
        if m_node in hin.node_type and d_node in hin.node_type:
            hin.add_edge(m_node, d_node, "has_director")

    return hin, ratings_df


# ─────────────────────────────────────────────────────────────────────────────
# Douban Movie  (13 367 users, 12 677 films)
# ─────────────────────────────────────────────────────────────────────────────

def load_douban_movie(data_dir: str = "data/douban_movie") -> tuple[HIN, pd.DataFrame]:
    """
    Load the Douban Movie dataset used in the paper.

    Expected files in *data_dir*:
        ratings.csv   — userId, movieId, rating
        movies.csv    — movieId, title, director   (optional: genres)
        social.csv    — userId1, userId2           (optional social links)

    Download from: https://github.com/S-Forouzandeh/RESCHet
    """
    ratings_path = os.path.join(data_dir, "ratings.csv")
    movies_path = os.path.join(data_dir, "movies.csv")

    if not os.path.exists(ratings_path):
        raise FileNotFoundError(f"ratings.csv not found in {data_dir}")

    ratings_df = pd.read_csv(ratings_path)

    hin = HIN()

    # users
    for uid in ratings_df["userId"].unique():
        hin.add_node(f"u_{uid}", "user")

    # movies + directors
    if os.path.exists(movies_path):
        movies_df = pd.read_csv(movies_path)
        for _, row in movies_df.iterrows():
            mid = row["movieId"]
            hin.add_node(f"m_{mid}", "movie")
            director = str(row.get("director", f"dir_{mid}"))
            d_node = f"d_{director}"
            if d_node not in hin.node_type:
                hin.add_node(d_node, "director")
            hin.add_edge(f"m_{mid}", d_node, "has_director")
    else:
        for mid in ratings_df["movieId"].unique():
            hin.add_node(f"m_{mid}", "movie")
            d_node = f"d_unknown"
            if d_node not in hin.node_type:
                hin.add_node(d_node, "director")
            hin.add_edge(f"m_{mid}", d_node, "has_director")

    # user–movie edges
    for row in ratings_df.itertuples(index=False):
        uid, mid = f"u_{row.userId}", f"m_{row.movieId}"
        if uid in hin.node_type and mid in hin.node_type:
            hin.add_edge(uid, mid, "rates")

    return hin, ratings_df


# ─────────────────────────────────────────────────────────────────────────────
# Yelp  (16 239 users, 14 282 businesses)
# ─────────────────────────────────────────────────────────────────────────────

def load_yelp(data_dir: str = "data/yelp") -> tuple[HIN, pd.DataFrame]:
    """
    Load the Yelp dataset used in the paper.

    Expected files in *data_dir*:
        ratings.csv   — userId, movieId (businessId), rating
        business.csv  — movieId, name, category

    Download from: https://github.com/S-Forouzandeh/RESCHet
    """
    ratings_path = os.path.join(data_dir, "ratings.csv")
    business_path = os.path.join(data_dir, "business.csv")

    if not os.path.exists(ratings_path):
        raise FileNotFoundError(f"ratings.csv not found in {data_dir}")

    ratings_df = pd.read_csv(ratings_path)
    hin = HIN()

    for uid in ratings_df["userId"].unique():
        hin.add_node(f"u_{uid}", "user")

    if os.path.exists(business_path):
        biz_df = pd.read_csv(business_path)
        for _, row in biz_df.iterrows():
            mid = row["movieId"]
            hin.add_node(f"m_{mid}", "movie")
            cat = str(row.get("category", "Unknown"))
            d_node = f"d_{cat}"
            if d_node not in hin.node_type:
                hin.add_node(d_node, "director")
            hin.add_edge(f"m_{mid}", d_node, "has_director")
    else:
        for mid in ratings_df["movieId"].unique():
            hin.add_node(f"m_{mid}", "movie")

    for row in ratings_df.itertuples(index=False):
        uid, mid = f"u_{row.userId}", f"m_{row.movieId}"
        if uid in hin.node_type and mid in hin.node_type:
            hin.add_edge(uid, mid, "rates")

    return hin, ratings_df
