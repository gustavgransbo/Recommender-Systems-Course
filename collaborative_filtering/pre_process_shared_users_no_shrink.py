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
parser.add_argument('output_name', metavar='output_name', type=str,
                    help='Nambe of the dataset being output')

args = parser.parse_args()
output_name = args.output_name

movielens_df = pd.read_csv('large_files/rating.csv')
movielens_df.drop(columns=['timestamp'])

# Re-label users and movies to range 0..n
user_ids = dict(zip(movielens_df.loc[:,'userId'].unique(), range(movielens_df.loc[:,'userId'].nunique())))
movie_ids = dict(zip(movielens_df.loc[:,'movieId'].unique(), range(movielens_df.loc[:,'movieId'].nunique())))

movielens_df.loc[:,'newUserId'] = movielens_df.loc[:,'userId'].apply(lambda x: user_ids[x])
movielens_df.loc[:,'newMovieId'] = movielens_df.loc[:,'movieId'].apply(lambda x: movie_ids[x])

movielens_df = movielens_df.sample(frac=1)
train_set_size = int(.8 * len(movielens_df))
train_df, test_df = movielens_df[:train_set_size], movielens_df[train_set_size:] 
print("Training set consists of %.2f %% of all reviews" % (100 * train_df.shape[0] / movielens_df.shape[0]))
print("Test set consists of %.2f %% of all reviews" % (100 * test_df.shape[0] / movielens_df.shape[0]))


train_df.to_csv('large_files/%s_shared_users_train.csv' % output_name, index=False)
test_df.to_csv('large_files/%s_shared_users_test.csv' % output_name, index=False)