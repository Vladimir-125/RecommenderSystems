import numpy as np
from scipy.sparse import rand as sprand
import torch
import pandas as pd

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# open training file
train   = pd.read_csv('./data/ratings_train.csv')
# open validation file
vali    = pd.read_csv('./data/ratings_vali.csv') 

# create user-movie matrix
user_movie = train.pivot('userId', 'movieId', 'rating')
# fill empty values with 0
user_movie =  user_movie.fillna(0.0)

# Make up some random explicit feedback ratings
# and convert to a numpy array
n_users = len(train.userId.unique())
n_items = len(train.movieId.unique())
userIds = list(user_movie.index)
movieIds = list(user_movie.columns)
# convert to numpy
ratings = user_movie.to_numpy()

class DenseNet(torch.nn.Module):

    def __init__(self, n_users, n_items):
        super().__init__()
   	    # user and item embedding layers
        factor_len = 30
        self.user_factors = torch.nn.Embedding(n_users, factor_len)
        self.item_factors = torch.nn.Embedding(n_items, factor_len)
   	    # linear layers
        self.inputs = torch.nn.Linear(factor_len*2, 50)
        # hidden liers
        self.linear1 = torch.nn.Linear(50, 30)
        self.linear2 = torch.nn.Linear(30, 20)
        # output lyer
        self.outputs = torch.nn.Linear(20, 1)

        self.to(DEVICE)

    def forward(self, users, items, dim):
        users_embedding = self.user_factors(users)
        items_embedding = self.item_factors(items)
	# concatenate user and item embeddings to form input
        x = torch.cat([users_embedding, items_embedding], dim)
        x = torch.relu(self.inputs(x))
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        output_scores = self.outputs(x)
        return output_scores


def batch(iterable1, iterable2, n=1):
    l = len(iterable1)
    for ndx in range(0, l, n):
        yield iterable1[ndx:min(ndx + n, l)], iterable2[ndx:min(ndx + n, l)]

# model instance
model = DenseNet(n_users, n_items)
model.load_state_dict(torch.load('./param.data'))
model.eval()

def train():
    # loss function
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2) # learning rate
    # Get indexes of nonzero elements (row indexes, col indexes)
    rows, cols = ratings.nonzero()
    # randomly shuffle array
    p = np.random.permutation(len(rows)) # returns shuffled indexes
    rows, cols = rows[p], cols[p]
    for epoch in range(50):
        loss_sum = 0
        batch_num = 1
        for row, col in batch(rows, cols, 256):
            optimizer.zero_grad()
            # Turn data into tensors
            rating = torch.FloatTensor(ratings[row, col]).to(DEVICE)
            #print(rating)
            row = torch.LongTensor([row]).to(DEVICE)
            col = torch.LongTensor([col]).to(DEVICE)
            
            # Predict and calculate loss
            prediction = model(row, col, 2)
            loss = loss_func(prediction.squeeze(), rating)
            # save total loss
            loss_sum += loss.item()
            batch_num += 1
            # Backpropagate
            loss.backward()
            # Update the parameters
            optimizer.step()
        print('Epoch: {}, loss: {}'.format(epoch+1, loss_sum/batch_num))

    torch.save(model.state_dict(), './param.data')

def predict(uid, mid):
    try:
        row = userIds.index(uid)
        col = movieIds.index(mid)
    except:
        return 'unknown'
    row = torch.LongTensor([row]).to(DEVICE)
    col = torch.LongTensor([col]).to(DEVICE)
    prediction = model(row, col, 1)
    return '{:.4f}'.format(prediction.item())


def read_user_id():
    with open('input.txt', 'r') as f:
        return [l.strip().split(',') for l in  f.readlines()]


def write_output(prediction):
    with open('output.txt', 'w') as f:
        for p in prediction:
                f.write(p + "\n")

def do(ids):
    # test implementation
    prediction = []
    for i in ids:
        rate1 = predict(int(i[0]), int(i[1]))
        prediction.append('{},{},{}'.format(i[0], i[1], rate1))
    return prediction

if __name__ == "__main__":
    user_ids = read_user_id()
    result = do(user_ids)
    write_output(result)
# train()