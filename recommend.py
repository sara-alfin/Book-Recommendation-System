"""
Kitap Öneri Sistemi (Book Recommendation System)

Book-Crossing veri seti üzerinde, kendi yazdığım NMF (Non-negative Matrix
Factorization) implementasyonu (bkz. nmf.py) ile çöküşümlü filtreleme
(collaborative filtering) tabanlı öneri üretir.

İki mod desteklenir:
  --mode book    : Kitap başlığına göre öneri (varsayılan)
  --mode author   : Yazara göre öneri

Kullanım
--------
    python recommend.py --user-id 178667 --mode book
    python recommend.py --user-id 178667 --mode author --init random

Bu dosya, projenin önceki sürümlerindeki (main.py, yazar_nmf*.py)
neredeyse birebir kopya olan dosyaların birleştirilmiş ve temizlenmiş
halidir.
"""

from __future__ import annotations

import argparse
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

import nmf


def read_data(
    users_path: str = "veri/Users.csv",
    books_path: str = "veri/Books.csv",
    ratings_path: str = "veri/Ratings.csv",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    users = pd.read_csv(users_path, sep=";", dtype=object)
    books = pd.read_csv(books_path, sep=";", dtype=object)
    ratings = pd.read_csv(ratings_path, sep=";", dtype=object)
    return users, books, ratings


def build_rating_matrix(
    mode: str,
    users_path: str = "veri/Users.csv",
    books_path: str = "veri/Books.csv",
    ratings_path: str = "veri/Ratings.csv",
    min_user_ratings: int = 250,
    min_item_ratings: int = 50,
    min_popular_ratings: int = 300,
) -> pd.DataFrame:
    """
    Kullanıcı x (kitap ya da yazar) puan matrisini (V) oluşturur.
    Ayrıca en popüler 100 kitap/yazarı `PopularBookRecommendation.pkl`
    dosyasına kaydeder.
    """
    group_key = "Title" if mode == "book" else "Author"

    users, books, ratings = read_data(users_path, books_path, ratings_path)
    ratings["Rating"] = ratings["Rating"].astype("int64")

    rating_with_name = ratings.merge(books, on="ISBN")

    num_rating_df = (
        rating_with_name.groupby(group_key).count()["Rating"].reset_index()
        .rename(columns={"Rating": "Num_rating"})
    )
    avg_rating_df = (
        rating_with_name.groupby(group_key).mean(numeric_only=True)["Rating"].reset_index()
        .rename(columns={"Rating": "Avg_rating"})
    )

    popular_df = num_rating_df.merge(avg_rating_df, on=group_key)
    popular_df = popular_df[popular_df["Num_rating"] >= min_popular_ratings]
    popular_df = popular_df.sort_values("Avg_rating", ascending=False).head(100)

    popular_df = popular_df.merge(books, on=group_key).drop_duplicates(group_key)[
        ["ISBN", "Title", "Author", "Publisher", "Num_rating", "Avg_rating"]
    ]
    pickle.dump(popular_df, open("PopularBookRecommendation.pkl", "wb"))

    active_users = rating_with_name.groupby("User-ID").count()["Rating"]
    active_users = active_users[active_users > min_user_ratings].index
    filtered = rating_with_name[rating_with_name["User-ID"].isin(active_users)]

    frequent_items = filtered.groupby(group_key).count()["Rating"]
    frequent_items = frequent_items[frequent_items >= min_item_ratings].index
    filtered = filtered[filtered[group_key].isin(frequent_items)]

    matrix = filtered.pivot_table(index=group_key, columns="User-ID", values="Rating")
    matrix.fillna(0, inplace=True)
    return matrix


def recommend(
    user_id: str,
    mode: str = "book",
    init_mode: str = "nndsvd",
    rank: int = 15,
    top_n: int = 10,
    users_path: str = "veri/Users.csv",
    books_path: str = "veri/Books.csv",
    ratings_path: str = "veri/Ratings.csv",
) -> pd.DataFrame:
    """Verilen kullanıcı için top-N öneriyi hesaplar ve DataFrame olarak döner."""
    group_key = "Title" if mode == "book" else "Author"

    df = build_rating_matrix(mode, users_path, books_path, ratings_path)
    _, books_master, _ = read_data(users_path, books_path, ratings_path)

    already_rated_mask = df.copy()
    already_rated_mask = already_rated_mask.ne(0)

    W, H = nmf.multiplicative_update(df, init_mode=init_mode, rank=rank)
    V = pd.DataFrame(np.dot(W, H), columns=df.columns, index=df.index)

    user_scores = V[user_id].sort_values(ascending=False)

    unread_titles = [
        item for item in user_scores.index
        if not already_rated_mask.at[item, user_id]
    ][:top_n]

    books_df = books_master.set_index(group_key)
    books_df.index = books_df.index.astype(str)
    books_df = books_df[~books_df.index.duplicated(keep="first")]

    display_columns = [c for c in ("ISBN", "Title", "Author") if c != group_key]
    recommendations = books_df.loc[unread_titles, display_columns]
    recommendations.insert(0, group_key, unread_titles)
    return recommendations.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="recommend",
        description="Book-Crossing veri seti üzerinde NMF tabanlı kitap/yazar öneri sistemi.",
    )
    parser.add_argument("-u", "--user-id", dest="user_id", type=str, default="178667",
                         help="Öneri üretilecek kullanıcı ID'si")
    parser.add_argument("-m", "--mode", dest="mode", choices=["book", "author"], default="book",
                         help="Öneri modu: 'book' (kitap başlığı) veya 'author' (yazar)")
    parser.add_argument("-i", "--init", dest="init_mode", choices=["random", "nndsvd"], default="nndsvd",
                         help="NMF başlatma yöntemi")
    parser.add_argument("--rank", dest="rank", type=int, default=15, help="NMF faktörizasyon rankı")
    parser.add_argument("--users-file", dest="users_path", default="veri/Users.csv")
    parser.add_argument("--books-file", dest="books_path", default="veri/Books.csv")
    parser.add_argument("--ratings-file", dest="ratings_path", default="veri/Ratings.csv")
    args = parser.parse_args()

    started_at = datetime.now()
    recommendations = recommend(
        user_id=args.user_id,
        mode=args.mode,
        init_mode=args.init_mode,
        rank=args.rank,
        users_path=args.users_path,
        books_path=args.books_path,
        ratings_path=args.ratings_path,
    )

    print(f"Kullanıcı {args.user_id} için önerilen {args.mode}lar:")
    print(recommendations)
    print(f"Tamamlanma süresi: {datetime.now() - started_at}")


if __name__ == "__main__":
    main()
