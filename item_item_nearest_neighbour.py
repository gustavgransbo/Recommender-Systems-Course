"""
Item-Item based collaborative filtering. 

Converted from the user-user script by chaning just a  few lines.
Mentions of users in this code often refer to items, I just did
not rename variables...
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
    train_df = pd.read_csv('large_files/movielens_large_few_movies_train.csv')
    test_df = pd.read_csv('large_files/movielens_large_few_movies_test.csv')
    
    # Set size variables
    n_users_train = train_df['userId'].nunique()
    n_users_test = test_df['userId'].nunique()
    n_movies = len(set(train_df['movieId'].unique()) & set(test_df['movieId'].unique())) # Hardcoded...

    print("Training users %d, Test users: %d, Movies: %d" % (n_users_train, n_users_test, n_movies))

    # Create sparse matrices (These are the only two lines that are changed from the user-user script)
    R_train = scipy.sparse.csr_matrix(sparse_from_df(train_df, n_users_train, n_movies).T)
    R_test = scipy.sparse.csr_matrix(sparse_from_df(test_df, n_users_test, n_movies).T)

    # Calculate user mean ratings
    R_train_user_mean = R_train.sum(1) / (R_train != 0).sum(1)
    R_test_user_mean = R_test.sum(1) / (R_test != 0).sum(1)

    # Calculate matrix of how ratings deviate from mean
    R_train_deviations = calculate_deviation_matrix(R_train, R_train_user_mean)
    R_test_deviations = calculate_deviation_matrix(R_test, R_test_user_mean)

    # Calculate user-user similarities (With item-item we only calculate item similarities based on training)
    W_train = cosine_similarity(R_train_deviations)

    # Evaluate on train and test set (Never allow an item to be considered a neighbour of it self: training_set=true)
    print("MSE on training set: %.4f" % predict_and_evaluate(R_train_deviations, R_train, R_train_user_mean, W_train, min(25, n_movies - 1), training_set=True))
    print("MSE on test set: %.4f" % predict_and_evaluate(R_test_deviations, R_test, R_test_user_mean, W_train, min(25, n_movies - 1), training_set=True))


"""
Example Outputs

With 10K training users and 10K test users:
train_df = pd.read_csv('large_files/movielens_small_train.csv')
test_df = pd.read_csv('large_files/movielens_small_test.csv')
100%|██████████████████████████████████████████████████████████████████████████| 2500/2500 [00:05<00:00, 475.67it/s]
MSE on training set: 0.6732
100%|██████████████████████████████████████████████████████████████████████████| 2500/2500 [00:04<00:00, 566.61it/s]
MSE on test set: 0.7265

With 1K traing and 1K test users:
train_df = pd.read_csv('large_files/movielens_very_small_train.csv')
test_df = pd.read_csv('large_files/movielens_very_small_test.csv')
100%|██████████████████████████████████████████████████████████████████████████| 2500/2500 [00:03<00:00, 706.17it/s]
MSE on training set: 0.6060
100%|██████████████████████████████████████████████████████████████████████████| 2500/2500 [00:03<00:00, 749.21it/s]
MSE on test set: 0.7534

Training users 25000, Test users: 10000, Movies: 20
100%|███████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 53.99it/s]
MSE on training set: 0.6363
100%|██████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 109.58it/s]
MSE on test set: 0.6353

Thoughts:
In all my experiments, user-user works better compared to item-item recommendations.
This is the opposite of what the course suggests.
I think I know hwy though! In my item-item based approach a allow no data leakage:
- Similarities between items are only based on the  training set data
- Items are not allowed to be considered similar to themselves
=> Predictions on the test set are in no way based on the true prediction a test set 
   user gave the movie being predicted.
For the training set, weights are calculated between test set users and traing set users
based on all ratings they have given. Including the predictions we are predicting.
This is a kind of data leakage, that could inflate the scores we get on the test set.
"""


