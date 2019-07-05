"""
This file implements matrix factorization for recommending movies to users.
It is a modified version of iterative_factorization.py, that includes biases
in the prediction.

The implementation approximates a user-item rating matrix R as
R_hat = WU' + user_bias + movie_bias + average_rating,
where R has dimensions NxM, W is NxK, U MxK, user_bias Nx1, 
movie_bias 1xM and average_rating is the average of R.
W and U, b and m are found by iteratively updating them as to minimize the mean
square error (MSE) of R_hat. 
"""
import scipy
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt

def sparse_from_df(df, n_users, n_movies):
    """
    Load a sparse recommendation matrix from a data frame containing user-rating pairs.
    """

    R = scipy.sparse.coo_matrix(
        (df.loc[:,'rating'].values, 
            (df.loc[:,'newUserId'].values, df.loc[:,'newMovieId'].values)
        ), shape=(n_users, n_movies))
    return scipy.sparse.csr_matrix(R)

def update_W_U(W, U, R):
    W = np.linalg.solve(U.T@U, U.T@R.T).T
    U = np.linalg.solve(W.T@W, W.T@R).T
    return W, U

def predict(user_id, movie_id, W, U, user_bias, movie_bias, average_rating):
    """
    Predict what rating a user gave to a specific movie.
    Important to make sure predictions are in the range [1, 5] 
    """
    r = W[user_id].dot(U[movie_id]) + user_bias[user_id] + movie_bias[movie_id] + average_rating
    if r < 1:
        return 1
    elif r > 5:
        return 5
    else:
        return r

def mse_eval(R, W, U, user_bias, movie_bias, avergae_rating):
    Y = R.data
    Y_hat = np.zeros(Y.shape)
    for idx, (i, j) in enumerate(zip(*R.nonzero())):
        Y_hat[idx] = predict(i, j, W, U, user_bias, movie_bias, average_rating)
    return mean_squared_error(Y, Y_hat)

def factorize_matrix(W, U, user_bias, movie_bias, average_rating, R, R_test, epochs = 50):

    mse_train_per_epoch = np.zeros(epochs)
    mse_test_per_epoch = np.zeros(epochs)
    pbar = tqdm(range(epochs))
    pbar.set_description("Test MSE: --")
    for epoch in pbar:
        
        # Update W
        for i in range(len(W)):
            available_indexes = R[i,:].nonzero()[1]
            u = U[available_indexes]
            r = np.squeeze(np.asarray(R[i,available_indexes].todense()))
            W[i] = np.linalg.solve(u.T.dot(u), (r - user_bias[i] - movie_bias[available_indexes] - average_rating).dot(u).T)
        # Update U
        for j in range(len(U)):
            available_indexes = R[:,j].nonzero()[0]
            r = np.squeeze(np.asarray(R[available_indexes,j].todense()))
            w = W[available_indexes]
            U[j] = np.linalg.solve(w.T.dot(w), (r - user_bias[available_indexes] - movie_bias[j] - average_rating).dot(w).T)
        
        # Update user_bias
        for i in range(len(W)):
            available_indexes = R[i,:].nonzero()[1]
            u = U[available_indexes]
            w = W[i]
            r = np.squeeze(np.asarray(R[i,available_indexes].todense()))
            user_bias[i] = (r - w.dot(u.T) - movie_bias[available_indexes] - average_rating).mean()
        
        # Update movie_bias
        for j in range(len(U)):
            available_indexes = R[:,j].nonzero()[0]
            r = np.squeeze(np.asarray(R[available_indexes,j].todense()))
            w = W[available_indexes]
            u = U[j]
            movie_bias[j] = (r - w.dot(u) - user_bias[available_indexes] - average_rating).mean()
        mse_train_per_epoch[epoch] = mse_eval(R, W, U, user_bias, movie_bias, average_rating)
        mse_test_per_epoch[epoch] = mse_eval(R_test, W, U, user_bias, movie_bias, average_rating)
        pbar.set_description("Test MSE: %.3f" % mse_test_per_epoch[epoch])
    return W, U, user_bias, movie_bias, mse_train_per_epoch, mse_test_per_epoch

if __name__ == "__main__":

    K = 10
    
    # Load data
    train_df = pd.read_csv('large_files/movielens_small_shared_users_train.csv')
    test_df = pd.read_csv('large_files/movielens_small_shared_users_test.csv')

    # Set size variables
    n_users = len(set(train_df['newUserId'].unique()).union(set(test_df['newUserId'].unique())))
    n_movies = len(set(train_df['newMovieId'].unique()) & set(test_df['newMovieId'].unique()))
    print("Users %d, Movies: %d" % (n_users, n_movies))

    # Create sparse matrices
    R_train = scipy.sparse.csr_matrix(sparse_from_df(train_df, n_users, n_movies))
    R_test = scipy.sparse.csr_matrix(sparse_from_df(test_df, n_users, n_movies))

    # Initialize factor matrices
    W = np.random.randn(n_users, K)
    U = np.random.randn(n_movies, K)

    # Initialize biases
    average_rating = R_train.sum() / (R_train != 0).sum()
    user_bias = np.random.randn(n_users)
    movie_bias = np.random.randn(n_movies)

    W, U, user_bias, movie_bias, train_results, test_results = factorize_matrix(
            W, U, user_bias, movie_bias, average_rating, R_train, R_test, epochs = 20)

    f, ax = plt.subplots(figsize=(16,9))
    ax.plot(train_results)
    ax.plot(test_results)
    ax.legend(['Train', 'Test'])
    ax.set_ylabel('MSE')
    ax.set_xlabel('epochs')
    ax.set_title('Matrix Factorization, Mean Squared Error (MSE)')
    plt.savefig('matrix_factorization/figures/iterative_factorization_with_bias.png')

    np.save('matrix_factorization/models/iterative_bias_W', W)
    np.save('matrix_factorization/models/iterative_bias_U', U)
    np.save('matrix_factorization/models/iterative_bias_user_bias', user_bias)
    np.save('matrix_factorization/models/iterative_bias_movie_bias', movie_bias)

"""
Results:
Train set MSE: 0.46927938
Test set MSE: 0.74801688

Thoughts:
Results are very close to what I got without biases (slightly worse),
but training now takes almost 3x as long. 
"""