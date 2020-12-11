# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

test_ratings = pd.read_csv('./data/ratings_test.csv')
train_ratings = pd.read_csv('./data/ratings_train.csv')

def get_movie_sim(train_ratings, fillMean=False):
    """ Pass trainig raitings that contains movieId, userId, rating columns
        Returns movie-movie similarity martix with movieId as indexes
        Cosine similarity is used
        Pass second argument as True if empty entries of sparce matrix should be filled by movie mean.
        Pass second argument as False if emoty entries of sparce matrix should be filled by 0."""
    movie_user = train_ratings.pivot('movieId', 'userId', 'rating')
    if fillMean:
        # set unrated movies to movie mean
        movie_means = movie_user.mean(axis=1)
        for i, col in enumerate(movie_user):
            # using i allows for duplicate columns
            movie_user.iloc[:, i] = movie_user.iloc[:, i].fillna(movie_means)
    else:
        # Fill NaNs with 0
        movie_user = movie_user.fillna(value=0)
    # Calculate cosine similarity
    # create normalazing vector
    norm = np.linalg.norm(movie_user, axis=1, keepdims=True)
    # normalize rates
    normalized_movie_repres = movie_user.div(norm)
    movie_sim = normalized_movie_repres.dot(normalized_movie_repres.T)
    # normalized_movie_repres = movie_user.dot(movie_user.T)
    # movie_sim = normalized_movie_repres.div(norm)
    return movie_sim

def get_movie_sim_v2(train_ratings, fillMean=False):
    """ Pass trainig raitings that contains movieId, userId, rating columns
        Returns movie-movie similarity martix with movieId as indexes
        Cosine similarity is used
        Pass second argument as True if empty entries of sparce matrix should be filled by movie mean.
        Pass second argument as False if emoty entries of sparce matrix should be filled by 0."""
    movie_user = train_ratings.pivot('movieId', 'userId', 'rating')
    if fillMean:
        # set unrated movies to movie mean
        movie_means = movie_user.mean(axis=1)
        for i, col in enumerate(movie_user):
            # using i allows for duplicate columns
            movie_user.iloc[:, i] = movie_user.iloc[:, i].fillna(movie_means)
    else:
        # Fill NaNs with 0
        movie_user = movie_user.fillna(value=0)
    # Calculate cosine similarity
    # create normalazing vector
    norm = np.linalg.norm(movie_user, axis=1, keepdims=True)
    # normalize rates
    # normalized_movie_repres = movie_user.div(norm)
    # movie_sim = normalized_movie_repres.dot(normalized_movie_repres.T)
    normalized_movie_repres = movie_user.dot(movie_user.T)
    movie_sim = normalized_movie_repres.div(norm)
    return movie_sim

def get_prediction(train_ratings, movie_sim, userId):
    """Returns predictions for a given user
        Requires: training ratings, movie_similarity, userId"""
    user_ratings = train_ratings[train_ratings.userId == userId]
    # retunr null of user does not exist in training set
    if user_ratings.empty:
        return user_ratings
    # get movie similarity user_rated_movieId x all_movies
    user_sim = movie_sim.loc[list(user_ratings.movieId),:]
    # drop if there are any mismatch
    user_sim = user_sim.dropna(how='any')
    # create pandas dataframe with 'movieId' as indexes and user ratings as 'rating' column
    user_ratings = pd.DataFrame({'rating': list(user_ratings.rating)},
                     index=user_ratings.movieId)
    # add one to ratings sum to prevent division by 0
    sim_sum = user_sim.sum() + 1
    # create pandas dataframe with 'movieId' as indexes and user ratings sum as 'rating' column
    sim_sum = pd.DataFrame({'rating': sim_sum.tolist()},
                          index=sim_sum.index)
    # multiply user_sim by user_ratings
    unnorm_ratings = user_sim.T.dot(user_ratings)
    # normalize user ratings
    user_all_movie_ratings = unnorm_ratings.div(sim_sum)
    # return user_rating predictions
    return user_all_movie_ratings

def calc_rmse(train_ratings, test_ratings, movie_sim, userId):
    """Calculate RMSE score for a single user
        Return: RMSE score for a user"""
    # get user predictions
    user_predicted_ratings = get_prediction(train_ratings, movie_sim, userId)
    # return None if unknown user
    if user_predicted_ratings.empty:
        return None
    # get user actual ratings
    test_user_ratings = test_ratings[test_ratings.userId == userId]
    
    # remove movies where predictions are not known
    unique = []
    for i in test_user_ratings.movieId:
        if i in movie_sim.index:
            unique.append(i)
    test_user_ratings = test_user_ratings[test_user_ratings.movieId.isin(unique)]
    # remove predictions that will not be used
    user_predicted_ratings = user_predicted_ratings[user_predicted_ratings.index.isin(test_user_ratings.movieId)]
    n = len(user_predicted_ratings)
    #err_square = (user_predicted_ratings.rating - test_user_ratings.rating)**2
    err_square = (np.array(user_predicted_ratings.rating) - np.array(test_user_ratings.rating))**2
    return (err_square.sum()/n)**(1/2)

def build_movie_repres(train_ratings):
    movie_user = train_ratings.pivot('movieId', 'userId', 'rating')
    movie_means = movie_user.mean(axis=1)
    for i, col in enumerate(movie_user):
        # using i allows for duplicate columns
        # inplace *may* not always work here, so IMO the next line is preferred
        # df.iloc[:, i].fillna(m, inplace=True)
        movie_user.iloc[:, i] = movie_user.iloc[:, i].fillna(movie_means)
    return movie_user
def get_factors(movie_user):
    u, s, vh = np.linalg.svd(movie_user, full_matrices=True)
    # take k factors
    K = 400
    U = u[:,:K]
    S = np.diag(s[:K])
    VH = vh[:K, :]
    P = U.dot(S)
    return P, VH

def get_prediction_svd(P, VH, movie_user, userId, movieId):
    if not int(userId) in list(movie_user.columns):
        print("Cannon predict for userId=" + str(userId))
        return 'unknown'
    elif not int(movieId) in list(movie_user.index):
        print("Cannot predict for movieId="+ str(movieId))
        return 'unknown'
    else:
        user = movie_user.columns.get_loc(int(userId))
        movie = movie_user.index.get_loc(int(movieId))
        user_predicted_ratings = P.dot(VH[:,user])
        return '{:.4f}'.format(user_predicted_ratings[movie])

def predict_movie_rate(train_ratings, movie_sim, userId, movieId):
    """Predict a rate for specific movieId"""
    user_recomendations = get_prediction(train_ratings, movie_sim, int(userId))
    if user_recomendations.empty:
        print("Cannon predict for userId=" + str(userId))
        return 'unknown'
    elif not int(movieId) in list(user_recomendations.index):
        print("Cannot predict for movieId="+ str(movieId))
        return 'unknown'
    rating = user_recomendations.loc[int(movieId)].rating
    return '{:.4f}'.format(rating)

def read_user_id():
    with open('input.txt', 'r') as f:
        return [l.strip().split(',') for l in  f.readlines()]


def write_output(prediction):
    with open('output.txt', 'w') as f:
        for p in prediction:
                f.write(p + "\n")

def do(ids):
    # test implementation
    movie_sim = get_movie_sim(train_ratings)
    movie_sim2 = get_movie_sim_v2(train_ratings)
    movie_user = build_movie_repres(train_ratings)
    P, VH = get_factors(movie_user)
    prediction = []
    for i in ids:
        rate1 = predict_movie_rate(train_ratings, movie_sim, i[0], i[1])
        prediction.append('{},{},{}'.format(i[0], i[1], rate1))
        rate2 = get_prediction_svd(P, VH, movie_user, i[0], i[1])
        prediction.append('{},{},{}'.format(i[0], i[1], rate2))
        rate3 = predict_movie_rate(train_ratings, movie_sim2, i[0], i[1])
        prediction.append('{},{},{}'.format(i[0], i[1], rate3))
    return prediction

if __name__ == "__main__":
    user_ids = read_user_id()
    result = do(user_ids)
    write_output(result)