import pandas as pd
import numpy as np

books = pd.read_csv('../data/books.csv')
ratings = pd.read_csv('../data/ratings.csv')
users = pd.read_csv('../data/users.csv')


def prepared_data(n):
    prep_books = books.drop(columns=["Publisher", "Image-URL-S", "Image-URL-M",
                                     "Image-URL-L"])

    user_ratings = ratings.groupby('User-ID')['Book-Rating'].count()
    book_ratings = ratings.groupby('ISBN')['Book-Rating'].count()

    filtered_users = user_ratings[user_ratings >= 100].index
    filtered_books = book_ratings[book_ratings >= 20].index
    filtered_ratings = ratings[ratings['User-ID'].isin(filtered_users) &
                               ratings['ISBN'].isin(filtered_books)]
    Y = filtered_ratings.pivot_table(index='ISBN', columns='User-ID',
                                     values='Book-Rating')
    R = Y.notna().astype(int)
    book_features = np.random.randn(Y.shape[0], n)
    user_features = np.random.randn(Y.shape[1], n)

    return prep_books, ratings, users, Y, R, user_features, book_features
