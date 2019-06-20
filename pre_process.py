# Script to pre-process and create a training and test set from the MovieLens20M data set

import pandas as pd
import numpy as np
np.random.seed(0)
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Pre-process MovieLens20M data set and split into train and test set.')
parser.add_argument('n_users_train', metavar='n_users_train', type=int,
                    help='Number of users to put in the training set')
parser.add_argument('n_users_test', metavar='n_users_test', type=int,
                    help='Number of users to put in the test set')
parser.add_argument('n_movies', metavar='n_movies', type=int,
                    help='Number of movies to consider for recommendations')
parser.add_argument('output_name', metavar='output_name', type=str,
                    help='Nambe of the dataset being output')

args = parser.parse_args()

n_users_train, n_users_test = args.n_users_train, args.n_users_test 
n_users = n_users_train + n_users_test
n_movies = args.n_movies
output_name = args.output_name

movielens_df = pd.read_csv('large_files/rating.csv')
movielens_df.drop(columns=['timestamp'])

# Only keep the movies with the most ratings
top_most_rated_movies = movielens_df['movieId'].value_counts()
top_movies = top_most_rated_movies[:n_movies].index.values
top_movies_df = movielens_df.loc[movielens_df.loc[:,'movieId'].isin(top_movies),:]
print("We are down to %.2f %% of all reviews" % (100 * top_movies_df.shape[0] / movielens_df.shape[0]))

# Keep a random subset of users
user_ids = top_movies_df['userId'].unique()
users_to_keep = np.random.choice(user_ids, n_users, replace=False)

# Split training and test set
train_users, test_users = users_to_keep[:n_users_train], users_to_keep[n_users_train:]
train_df = top_movies_df.loc[top_movies_df.loc[:,'userId'].isin(train_users),:].copy()
test_df = top_movies_df.loc[top_movies_df.loc[:,'userId'].isin(test_users),:].copy()
print("Training set consists of %.2f %% of all reviews" % (100 * train_df.shape[0] / movielens_df.shape[0]))
print("Test set consists of %.2f %% of all reviews" % (100 * test_df.shape[0] / movielens_df.shape[0]))

# Re-label users and movies to range 0..n
train_user_ids = dict(zip(train_df.loc[:,'userId'].unique(), range(train_df.loc[:,'userId'].nunique())))
test_user_ids = dict(zip(test_df.loc[:,'userId'].unique(), range(test_df.loc[:,'userId'].nunique())))
movie_ids = dict(zip(top_movies, range(len(top_movies))))

# Relabel Users
train_df.loc[:,'newUserId'] = train_df.loc[:,'userId'].apply(lambda x: train_user_ids[x])
test_df.loc[:,'newUserId'] = test_df.loc[:,'userId'].apply(lambda x: test_user_ids[x])

# Relabel Movies
train_df.loc[:,'newMovieId'] = train_df.loc[:,'movieId'].apply(lambda x: movie_ids[x])
test_df.loc[:,'newMovieId'] = test_df.loc[:,'movieId'].apply(lambda x: movie_ids[x])

train_df.to_csv('large_files/%s_train.csv' % output_name, index=False)
test_df.to_csv('large_files/%s_test.csv' % output_name, index=False)