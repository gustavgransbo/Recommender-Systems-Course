"""
Multi Layer Perceptron based movie recommendations using Keras.
Parts of this code was heavily insipred by:
https://github.com/lazyprogrammer/machine_learning_examples/blob/master/recommenders/mf_keras.py
"""

import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Add, Dot, Embedding, Input, Flatten, Concatenate, Dense
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.backend import squeeze
import time
import matplotlib.pyplot as plt

def setup_model(N, M, K, reg = .0):

    # Input
    user = Input(shape=(1,))
    movie = Input(shape=(1,))

    # Parameters
    w = Embedding(N, K, embeddings_regularizer=l2(reg))(user) # (B, 1, K)
    u = Embedding(M, K, embeddings_regularizer=l2(reg))(movie) # (B, 1, K)
    w = Flatten()(w) # (B, K)
    u = Flatten()(u) # (B, K)

    # Estimate Rating
    x = Concatenate(axis=-1)([w, u]) # (B, 2*K)
    x = Dense(units=32, input_shape=(2*K,), activation='tanh')(x)
    x = Dense(units=1, input_shape=(32,))(x)


    return Model(inputs=[user, movie], outputs=x)

if __name__ == "__main__":

    # Hidden dimensions
    K = 10

    # Load data
    train_df = pd.read_csv('large_files/movielens_small_shared_users_train.csv')
    test_df = pd.read_csv('large_files/movielens_small_shared_users_test.csv')

    # Center Ratings
    average_rating = train_df['rating'].mean()
    train_df['rating'] -= average_rating
    test_df['rating'] -= average_rating

    # Set size variables
    n_users = len(set(train_df['newUserId'].unique()).union(set(test_df['newUserId'].unique())))
    n_movies = len(set(train_df['newMovieId'].unique()) & set(test_df['newMovieId'].unique()))
    print("Users %d, Movies: %d" % (n_users, n_movies))

    X_train = train_df[['newUserId', 'newMovieId']].values
    X_test = test_df[['newUserId', 'newMovieId']].values
    y_train = train_df['rating'].values
    y_test = test_df['rating'].values

    mf = setup_model(n_users, n_movies, K, reg = 1e-7)

    mf.compile(
        loss = 'mse',
        optimizer = SGD(lr = .01, momentum=.9),
        metrics = ['mse']
    )

    t1 = time.time()
    res = mf.fit(
        x = [X_train[:,0], X_train[:,1]],
        y = y_train,
        epochs = 20,
        batch_size=128,
        validation_data=([X_test[:,0], X_test[:,1]], y_test)
    )
    
    train_results = res.history['mean_squared_error']
    test_results=  res.history['val_mean_squared_error']
    print("Total Training Time %.2f s" % (time.time() - t1))
    print("Train MSE: %.5f" % train_results[-1])
    print("Test MSE: %.5f" % test_results[-1])

    f, ax = plt.subplots(figsize=(16,9))
    ax.plot(train_results)
    ax.plot(test_results)
    ax.legend(['Train', 'Test'])
    ax.set_ylabel('MSE')
    ax.set_xlabel('epochs')
    ax.set_title('Matrix Factorization, Mean Squared Error (MSE)')
    plt.savefig('matrix_factorization/figures/mlp_keras_reg_small.png')

"""
Results
Without regularization
>Users 10000, Movies: 2500
>Train MSE: 0.65623
>Test MSE: 0.68431
~ 10 minutes

Comment:
Better than the simple Keras MF implementation, and almost on par with the 
regularized alternating least squares implementation in terms of test performance
(still slightly worse).

With regularization
>Users 10000, Movies: 2500
>Total Training Time 1593.77 s
>Train MSE: 0.65647
>Test MSE: 0.68265

Comment:
Very similar results to the experiment without regularization.
It might be that I failed at selecting a useful regularization parameter.
"""