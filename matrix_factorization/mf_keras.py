"""
Implementing matrix factorization based movie recommendations using Keras.
Parts of this code was heavily insipred by:
https://github.com/lazyprogrammer/machine_learning_examples/blob/master/recommenders/mf_keras.py
"""

import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Add, Dot, Embedding, Input, Flatten
from keras.regularizers import l2
from keras.optimizers import SGD

def setup_model(N, M, K, reg = .0):

    # Input
    user = Input(shape=(1,))
    movie = Input(shape=(1,))

    # Parameters
    w = Embedding(N, K, embeddings_regularizer=l2(reg))(user) # (B, 1, K)
    u = Embedding(M, K, embeddings_regularizer=l2(reg))(movie) # (B, 1, K)
    user_bias = Embedding(N, 1, embeddings_regularizer=l2(reg))(user) # (B, 1, 1)
    movie_bias = Embedding(M, 1, embeddings_regularizer=l2(reg))(movie) # (B, 1, 1)

    # Estimate Rating
    x = Dot(axes=2)([w, u]) # (B, 1, 1)
    x = Add()([x, user_bias, movie_bias])
    x = Flatten()(x)

    return Model(inputs=[user, movie], outputs=x)

if __name__ == "__main__":

    # Hidden dimensions
    K = 10

    # Load data
    train_df = pd.read_csv('large_files/movielens_larger_shared_users_train.csv')
    test_df = pd.read_csv('large_files/movielens_larger_shared_users_test.csv')

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

    mf = setup_model(n_users, n_movies, K)

    mf.compile(
        loss = 'mse',
        optimizer = SGD(lr = .01, momentum=.9),
        metrics = ['mse']
    )

    res = mf.fit(
        x = [X_train[:,0], X_train[:,1]],
        y = y_train,
        epochs = 20,
        batch_size=128,
        validation_data=([X_test[:,0], X_test[:,1]], y_test)
    )

"""
Results
Without regularization
>Users 10000, Movies: 2500
>Epoch 20/20
>980032/980032 [==============================] - 30s 31us/step - loss: 0.7095 - 
>mean_squared_error: 0.7095 - val_loss: 0.7275 - val_mean_squared_error: 0.7275

Comment:
Faster than my PyTorch implementation. Almost 2x as fast.
Also, this implementation got a little bit better results in terms of test MSE.
Keras does regularization similar to PyTorch (all parameters every batch),
so I did not get regularization to work well here either.

>Users 25000, Movies: 2500
>Epoch 20/20
>2443069/2443069 [==============================] - 90s 37us/step - loss: 0.7140 - 
>mean_squared_error: 0.7140 - val_loss: 0.7297 - val_mean_squared_error: 0.7297

Comment:
Slightly faster than my alternating least squares implementation at processing the larger
data set. Unfortunately with worse results, probably since I have no regularization...
Though, it is weird that even without regularization the training loss does not improve more.
"""