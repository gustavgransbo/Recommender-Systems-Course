"""
This file implements matrix factorization for recommending movies to users.

The implementation approximates a user-item rating matrix R as R_hat = WU',
where R has dimensions NxM, W is NxK and U MxK. 
W and U are found by iteratively updating them as to minimize the mean
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

def predict(user_id, movie_id, W, U):
    """
    Predict what rating a user gave to a specific movie.
    Important to make sure predictions are in the range [1, 5] 
    """
    r = W[user_id].dot(U[movie_id])
    if r < 1:
        return 1
    elif r > 5:
        return 5
    else:
        return r

def mse_eval(R, W, U):
    Y = R.data
    Y_hat = np.zeros(Y.shape)
    for idx, (i, j) in enumerate(zip(*R.nonzero())):
        Y_hat[idx] = predict(i, j, W, U)
    return mean_squared_error(Y, Y_hat)

def factorize_matrix(W, U, R, R_test, epochs = 50):

    mse_train_per_epoch = np.zeros(epochs)
    mse_test_per_epoch = np.zeros(epochs)
    for epoch in tqdm(range(epochs)):
        # Update W
        for i in range(len(W)):
            available_indexes = R[i,:].nonzero()[1]
            u = U[available_indexes]
            r = np.squeeze(np.asarray(R[i,available_indexes].todense()))
            W[i] = np.linalg.solve(u.T.dot(u), r.dot(u).T)
        # Update U
        for j in range(len(U)):
            available_indexes = R[:,j].nonzero()[0]
            r = np.squeeze(np.asarray(R[available_indexes,j].todense()))
            w = W[available_indexes]
            U[j] = np.linalg.solve(w.T.dot(w), r.dot(w).T)
        mse_train_per_epoch[epoch] = mse_eval(R, W, U)
        mse_test_per_epoch[epoch] = mse_eval(R_test, W, U)
    return W, U, mse_train_per_epoch, mse_test_per_epoch

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

    W, U, train_results, test_results = factorize_matrix(W, U, R_train, R_test, epochs = 20)

    f, ax = plt.subplots(figsize=(16,9))
    ax.plot(train_results)
    ax.plot(test_results)
    ax.legend(['Train', 'Test'])
    ax.set_ylabel('MSE')
    ax.set_xlabel('epochs')
    ax.set_title('Matrix Factorization, Mean Squared Error (MSE)')
    plt.savefig('matrix_factorization/figures/iterative_factorization.png')

    np.save('matrix_factorization/models/iterative_W', W)
    np.save('matrix_factorization/models/iterative_U', U)

"""
Results:
Train set MSE: 0.46645975
Test set MSE: 0.73397646

Thoughts:
MSE on the test set is a little bit better than what I got using collabortive filtering
MSE on the training set is much better, but that is not very relevant. The reason is
most likely that the collaborative filtering approach was not able to peek as directly
at the training data as the matrix factorization method is able to do.

Training is very slow 10 minutes for 20 epochs on my laptop. This was expected
since I am calculating W_i and U_i in Python for-loops. Using a matrix factorization
package would be much more ideal, but this was a nice learning experience.
"""