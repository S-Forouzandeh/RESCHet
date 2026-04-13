"""
main.py — RESCHet demo on MovieLens (ml-latest-small)

Usage
─────
    # 1. Download MovieLens
    mkdir -p data
    wget https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
    unzip ml-latest-small.zip -d data/

    # 2. Run
    python main.py
"""

import argparse
import sys
import pandas as pd

from data_utils import load_movielens, load_douban_movie, load_yelp
from reschet import RESCHet


def build_user_item_ratings(
    ratings_df: pd.DataFrame,
    user_col: str = "userId",
    item_col: str = "movieId",
    rating_col: str = "rating",
) -> dict:
    """Convert a ratings DataFrame to {u_id: {m_id: rating}} for recommend()."""
    ui: dict = {}
    for row in ratings_df.itertuples(index=False):
        uid = f"u_{getattr(row, user_col)}"
        mid = f"m_{getattr(row, item_col)}"
        ui.setdefault(uid, {})[mid] = float(getattr(row, rating_col))
    return ui


def run(args):
    # ── 1. Load dataset ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RESCHet — dataset: {args.dataset}")
    print(f"{'='*60}\n")

    if args.dataset == "movielens":
        print("Loading MovieLens dataset …")
        hin, ratings_df = load_movielens(
            args.data_dir, max_ratings=args.max_ratings
        )
    elif args.dataset == "douban_movie":
        print("Loading Douban Movie dataset …")
        hin, ratings_df = load_douban_movie(args.data_dir)
    elif args.dataset == "yelp":
        print("Loading Yelp dataset …")
        hin, ratings_df = load_yelp(args.data_dir)
    else:
        print(f"Unknown dataset '{args.dataset}'. Choose: movielens | douban_movie | yelp")
        sys.exit(1)

    print("HIN statistics:")
    hin.info()

    # ── 2. Initialise & train RESCHet ────────────────────────────────────────
    print("\nInitialising RESCHet …")
    model = RESCHet(
        embedding_dim=args.embedding_dim,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
        window=args.window,
        max_k=args.max_k,
        top_n=args.top_n,
        workers=args.workers,
    )
    model.fit(hin)

    # ── 3. Sample recommendations ────────────────────────────────────────────
    user_item_ratings = build_user_item_ratings(ratings_df)
    sample_users = model.user_ids[:3]

    print("Sample recommendations")
    print("-" * 40)
    for user in sample_users:
        recs = model.recommend(user, user_item_ratings, top_n=args.top_n)
        already = list(user_item_ratings.get(user, {}).keys())[:3]
        print(f"  {user}")
        print(f"    Previously rated : {already}")
        print(f"    Recommended      : {recs[:5]}")
    print()

    # ── 4. Evaluation (MAE / RMSE) ───────────────────────────────────────────
    if args.evaluate:
        print("Evaluating (MAE / RMSE across train ratios) …")
        if args.dataset == "yelp":
            train_ratios = [0.6, 0.7, 0.8, 0.9]
        else:
            train_ratios = [0.2, 0.4, 0.6, 0.8]

        results = model.evaluate(ratings_df, train_ratios=train_ratios)
        print("\nEvaluation Results")
        print("=" * 60)
        print(results.to_string(index=False))
        print()

        if args.save_results:
            out_path = f"results_{args.dataset}.csv"
            results.to_csv(out_path, index=False)
            print(f"Results saved → {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="RESCHet demo")
    p.add_argument("--dataset", default="movielens",
                   choices=["movielens", "douban_movie", "yelp"])
    p.add_argument("--data_dir", default="data/ml-latest-small")
    p.add_argument("--max_ratings", type=int, default=30_000,
                   help="Max ratings to load for MovieLens (speed)")
    p.add_argument("--embedding_dim", type=int, default=64)
    p.add_argument("--walk_length", type=int, default=20)
    p.add_argument("--num_walks", type=int, default=3)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--max_k", type=int, default=10)
    p.add_argument("--top_n", type=int, default=10)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--evaluate", action="store_true",
                   help="Run MAE/RMSE evaluation")
    p.add_argument("--save_results", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
