# Script to pre-process and create a training and test set from the MovieLens20M data set
# Difference to the pre_process.py is that this script allows users to appear in both
# the test and training set, just like they do in the course! 
# I think this way of splitting the data up actually makes a lot more sense,
# also, it removed the problem of leakage in the user-user approach.
#
# I want to see how this affects my results. 
# (I will also need to rewrite the collaborative filtering scripts a little bit) 

import pandas as pd
import numpy as np
np.random.seed(0)
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Pre-process MovieLens20M data set and split into train and test set.')
parser.add_argument('n_users', metavar='n_users_train', type=int,
                    help='Number of users to put keep')
parser.add_argument('n_movies', metavar='n_movies', type=int,
                    help='Number of movies to consider for recommendations')
parser.add_argument('output_name', metavar='output_name', type=str,
                    help='Nambe of the dataset being output')

args = parser.parse_args()

n_users = args.n_users
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

# Keep only selected users
reduced_df = top_movies_df.loc[top_movies_df.loc[:,'userId'].isin(users_to_keep),:].copy()

# Re-label users and movies to range 0..n
user_ids = dict(zip(reduced_df.loc[:,'userId'].unique(), range(reduced_df.loc[:,'userId'].nunique())))
movie_ids = dict(zip(top_movies, range(len(top_movies))))

reduced_df.loc[:,'newUserId'] = reduced_df.loc[:,'userId'].apply(lambda x: user_ids[x])
reduced_df.loc[:,'newMovieId'] = reduced_df.loc[:,'movieId'].apply(lambda x: movie_ids[x])

reduced_df = reduced_df.sample(frac=1)
train_set_size = int(.8 * len(reduced_df))
train_df, test_df = reduced_df[:train_set_size], reduced_df[train_set_size:] 
print("Training set consists of %.2f %% of all reviews" % (100 * train_df.shape[0] / movielens_df.shape[0]))
print("Test set consists of %.2f %% of all reviews" % (100 * test_df.shape[0] / movielens_df.shape[0]))


train_df.to_csv('large_files/%s_shared_users_train.csv' % output_name, index=False)
test_df.to_csv('large_files/%s_shared_users_test.csv' % output_name, index=False)