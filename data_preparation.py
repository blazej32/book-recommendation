import pandas as pd
import numpy as np

books = pd.read_csv('./data/books.csv')
ratings = pd.read_csv('./data/ratings.csv')
users = pd.read_csv('./data/users.csv')


def prepared_data():
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
    return prep_books, ratings, users, Y.values, R.values


def create_ratings_array(ratings_dict, y):
    num_books = y.shape[0]
    ratings_array = np.empty((num_books,))

    for i in range(num_books):
        isbn = books[i]['isbn']
        if isbn in ratings_dict:
            ratings_array[i] = ratings_dict[isbn]
        else:
            ratings_array[i] = np.nan

    return ratings_array
