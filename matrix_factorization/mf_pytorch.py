"""
Implementing matrix factorization based movie recommendations using PyTorch
"""
import torch
import torch.nn as nn
import pandas as pd
import torch.utils.data
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error

class MF(nn.Module):
    def __init__(self, N, M, K):
        super(MF, self).__init__()
        self.N = N
        self.M = M
        self.K = K

        self.W = nn.Embedding(N, K)
        self.U = nn.Embedding(M, K)
        self.user_bias = nn.Embedding(N, 1)
        self.movie_bias = nn.Embedding(M, 1)

    def forward(self, inputs):
        """
        inputs (Bx2): [
            [user_id, movie_id],
            [user_id, movie_id],
            ...
        ] Size: (batch_size, 2)
        """
        users = inputs[:,0]
        movies = inputs[:,1]
        w = self.W(users)
        u = self.U(movies)
        batch_size = inputs.size(0)
        x = torch.bmm(w.view(batch_size, 1, self.K), u.view(batch_size, self.K, 1)).view(-1, 1)
        return (x + self.user_bias(users) + self.movie_bias(movies)).view(-1)


# Function inspired by https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def train_model(model, train_loader, optimizer, criterion, validation_loader = None, epochs = 2):
    
    # Only enter the validation state if there is a validation_loader
    phases = ['train']
    data_set_loaders = {'train' : train_loader, 'val' : validation_loader} 
    if validation_loader:
        phases.append('val')
        
    for epoch in range(epochs):
        
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        for phase in phases:
            
            data_set_loader = data_set_loaders[phase]
            
            # Only update model weights based on the training data
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = .0
            running_mse = .0
            
            for _, batch in tqdm(enumerate(data_set_loader), 
                                 total = int(np.ceil(len(data_set_loader.dataset) / data_set_loader.batch_size))):
                inputs, labels = batch
                
                optimizer.zero_grad()
                
                # Only track history during training
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    mse = mean_squared_error(labels, outputs.detach())
                    
                    # Only perform backpropagation during training
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                # Save statistics
                running_loss += loss.item() * inputs.size(0)
                running_mse += mse * inputs.size(0)
                
            epoch_loss = running_loss / len(data_set_loader.dataset)
            epoch_mse = running_mse / len(data_set_loader.dataset)

            print('{} Loss: {:.4f} MSE: {:.4f}'.format(
                phase, epoch_loss, epoch_mse))

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

    X_train = torch.from_numpy(train_df[['newUserId', 'newMovieId']].values)
    X_test = torch.from_numpy(test_df[['newUserId', 'newMovieId']].values)
    y_train = torch.from_numpy(train_df['rating'].values).float()
    y_test = torch.from_numpy(test_df['rating'].values).float()

    batch_size = 256
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    mf = MF(n_users, n_movies, K)
    optimizer = torch.optim.SGD(mf.parameters(), lr=0.1, momentum=.9)
    criterion = nn.MSELoss()

    train_model(mf, train_loader, optimizer, criterion, validation_loader = test_loader, epochs = 10 )

"""
Results:

Without regularization:
>Users 10000, Movies: 2500
...
>Epoch 10/10
>----------
>100%|███████████████████████████████████████████████████████████████████████████| 3829/3829 [00:57<00:00, 66.52it/s]
>train Loss: 0.7103 MSE: 0.7103
>100%|████████████████████████████████████████████████████████████████████████████| 479/479 [00:04<00:00, 107.51it/s]
>val Loss: 0.7770 MSE: 0.7770

Comment:
Code runs a little bit slower compared to the alternatig squares implementation,
and the test MSE is a little bit worse. I bet I got something wrong in my implementation.
I will try to veryify by also implementing the code i Keras.

"""
