"""
This script implements my neighbour based user-user solution 
from  the notebook "User-User Collaborative Filtering"

The purpose of the script is to easier be able to re-run the 
code on other train-test splits.
"""

import pandas as pd
import numpy as np
import scipy
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def sparse_from_df(df, n_users, n_movies):
    R = scipy.sparse.coo_matrix(
        (df.loc[:,'rating'].values, 
            (df.loc[:,'newUserId'].values, df.loc[:,'newMovieId'].values)
        ), shape=(n_users, n_movies))
    return scipy.sparse.csr_matrix(R)

def calculate_deviation_matrix(R, R_user_means):
    # Elementwise subtraction of the user means (adapted from https://stackoverflow.com/questions/19017804/scipy-sparse-matrix-special-substraction)
    R_user_mean_vec = np.asarray(R_user_means).reshape(-1)
    
    non_zero_per_row = np.diff(R.indptr)
    return scipy.sparse.csr_matrix(
                (R.data - np.repeat(R_user_mean_vec, non_zero_per_row), R.indices, R.indptr),
                shape=R.shape)

def pred_user_from_k_neighbours(user_id, R_deviations, W, R_user_mean, k, training_set=False):
    
    if training_set:
        # If we are preicting in the training set, don't consider a user to be a neighbour of herself
        k_closest_users = np.argpartition(W[user_id], -k-1)[-k-1:]
        k_closest_users = np.delete(k_closest_users, np.where(k_closest_users == user_id))
    else:
        k_closest_users = np.argpartition(W[user_id], -k)[-k:]
    
    weighted_deviations = R_deviations[k_closest_users].T.multiply(W[user_id,k_closest_users]).T
    
    return R_user_mean[user_id] + np.nan_to_num((weighted_deviations.sum(0) / np.absolute(W[user_id,k_closest_users]).sum(0)), 0)

def predict_and_evaluate(R_train_deviations, R_test, R_test_user_means, W_test, k_neighbours, training_set=False):

    # Don't print warning when dividing by zero, I'm adressing that right after
    np.seterr(divide='ignore', invalid='ignore')    

    predictions = np.zeros(R_test.shape)
    for i in tqdm(range(R_test.shape[0])):
        predictions[i,:] = pred_user_from_k_neighbours(i, R_train_deviations, W_test, R_test_user_means, k_neighbours, training_set)

    np.seterr(divide='warn', invalid='warn')   

    # We can only evaluate predictions we have a right answer to
    Y = R_test.data
    Y_hat = np.asarray(predictions[R_test.nonzero()]).reshape(-1)


    return mean_squared_error(Y, Y_hat)

if __name__ == "__main__":
    
    # Load data
    train_df = pd.read_csv('large_files/movielens_small_train.csv')
    test_df = pd.read_csv('large_files/movielens_small_test.csv')
    
    # Set size variables
    n_users_train = train_df['userId'].nunique()
    n_users_test = test_df['userId'].nunique()
    n_movies = 2500 # Hardcoded...

    # Create sparse matrices
    R_train = sparse_from_df(train_df, n_users_train, n_movies)
    R_test = sparse_from_df(test_df, n_users_test, n_movies)

    # Calculate user mean ratings
    R_train_user_mean = R_train.sum(1) / (R_train != 0).sum(1)
    R_test_user_mean = R_test.sum(1) / (R_test != 0).sum(1)

    # Calculate matrix of how ratings deviate from mean
    R_train_deviations = calculate_deviation_matrix(R_train, R_train_user_mean)
    R_test_deviations = calculate_deviation_matrix(R_test, R_test_user_mean)

    # Calculate user-user similarities
    W_train = cosine_similarity(R_train_deviations)
    W_test = cosine_similarity(R_test_deviations, R_train_deviations)

    # Evaluate on train and test set
    print("MSE on training set: %.4f" % predict_and_evaluate(R_train_deviations, R_train, R_train_user_mean, W_train, 25, training_set=True))
    print("MSE on test set: %.4f" % predict_and_evaluate(R_train_deviations, R_test, R_test_user_mean, W_test, 25, training_set=False))


"""
Example Outputs

With 10K training users and 10K test users:
train_df = pd.read_csv('large_files/movielens_small_train.csv')
test_df = pd.read_csv('large_files/movielens_small_test.csv')
> 100%|████████████████████████████████████████████████████████████████████████| 10000/10000 [00:13<00:00, 729.21it/s]
> MSE on training set: 0.6442
> 100%|████████████████████████████████████████████████████████████████████████| 10000/10000 [00:12<00:00, 776.10it/s]
> MSE on test set: 0.6561

With 1K traing and 1K test users:
train_df = pd.read_csv('large_files/movielens_very_small_train.csv')
test_df = pd.read_csv('large_files/movielens_very_small_test.csv')
100%|██████████████████████████████████████████████████████████████████████████| 1000/1000 [00:02<00:00, 454.93it/s]
MSE on training set: 0.7153
100%|██████████████████████████████████████████████████████████████████████████| 1000/1000 [00:02<00:00, 458.89it/s]
MSE on test set: 0.6873
"""


