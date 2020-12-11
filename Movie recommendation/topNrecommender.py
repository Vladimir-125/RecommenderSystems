import numpy as np
import pandas as pd

movies   = pd.read_csv('./data/movies_w_imgurl.csv')
tags_csv = pd.read_csv('./data/tags.csv')

# count number of movies
# count number of unique id
movie_count = {}
# how many times each unique tag has occured in all documents
unique_tags = {} 
for i in range(len(tags_csv)):
    # sepatare string into tags
    tags = tags_csv['tag'][i].split(',')
    movie = str(tags_csv['movieId'][i])
    for tag in tags:
        tag = tag.strip()
        unique_tags.setdefault(tag, 0)
        unique_tags[tag] +=1
        
        movie_count.setdefault(movie, [])
        movie_count[movie].append(tag)

def count_IDF():
    'count IDF for each tag'
    IDF = {}
    # total number of movies with tags
    total_movie_count = len(movie_count)
    for tag in unique_tags.keys():
        IDF[tag] = np.log10(total_movie_count/unique_tags[tag])
    return IDF

def create_TF():
    'count TF for each movie in tags csv'
    TF = {}
    for movie in movie_count.keys():
        row = {}
        for tag in unique_tags.keys():
            total_tags = len(movie_count[movie])
            tags_inmovie = 0
            for m_tag in movie_count[movie]:
                # count how many times current tag appeared at the movie
                if tag == m_tag:
                    tags_inmovie+=1
            row[tag] = tags_inmovie/total_tags
        TF[movie] = row
    return TF

def tags_representation(): 
    'returns movieID*tag TF-IDF representation'
    IDF   = count_IDF()
    TF    = create_TF()
    TFIDF = {}
    for movie in TF:
        row = {}
        for tag in TF[movie]:
            row[tag] = TF[movie][tag] * IDF[tag]

        TFIDF[movie] = row
    
    return TFIDF

def jenres_representation(movies):
    'Returns genres movie representation'
    movie_representation = {} # final movie representation
    total_count = len(movies)
    genre_count = {}
    
    for genres in movies.genres:
        genre_list = genres.split('|')
        for genre in genre_list:
            genre_count.setdefault(genre, 0) # create new element if not exist
            genre_count[genre]+=1 # increment if exist

    genre_list = list(genre_count.keys()) # create a list of keys
    genre_list.sort()
    dict_movies = dict(movies)
    
    for i in  range(len(dict_movies['movieId'])):
        row = {}
        for g in genre_list:
            if g in dict_movies['genres'][i].split('|'):
                IDF = np.log10((total_count/genre_count[g]))
                row[g] = IDF
            else:
                row[g] = 0
        movie_representation[str(dict_movies['movieId'][i])] = row
        
    return movie_representation  

def final_representation():
    'Returns movieId*(jenres+tags) representation'
    movie_repres = jenres_representation(movies)
    tag_repres = tags_representation()
    # list of tagged movieIds
    tag_movieIds = list(tag_repres.keys())
    # list of tags
    tags = list(tag_repres[tag_movieIds[0]].keys())
    # initiate new tags to 0
    for movie in movie_repres:
        for tag in tags:
            # check if movie already has genre=tag
            if tag in movie_repres[movie].keys():
                # rename old genre to old_genre+randint (mydict[new_key] = mydict.pop(old_key))
                movie_repres[movie][tag+str(np.random.randint(999999))] =  movie_repres[movie][tag]
            movie_repres[movie][tag] = 0.0
    
    # add movie tags to genre representation
    for movieId in tag_movieIds:
        for tag in tags:
             movie_repres[movieId][tag] = tag_repres[movieId][tag]
    return movie_repres

def movie_similarity():
    'Finds cosine similarity between movies'
    movie_repres = final_representation()
    array2d = []
    # dict to 2d array movie representation
    for movie in movie_repres:
        row = list(movie_repres[movie].values())
        array2d.append(row)
    
    # numpy matrix movie representation
    mat_movie_repres = np.matrix(array2d)
    return mat_movie_repres

# def mat_mat_cos_similarity(mat):
#     result = []
#     print('Started calculating movie-movie cosine similarity...')
#     for rows in range(len(mat)):
#         row = []
#         for cols in range(len(mat)):
#             a = mov_sim[rows,:]
#             b = mov_sim.T[:,cols]
#             cosine = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
#             row.append(float(cosine))
#         result.append(row)
#         if rows%100==0:
#             print(str(round(rows*100/len(mat), 2)) + '% is done.')
#             # save intermidiate movie similarity
#             save_obj(np.matrix(result), 'movie-cos-similarity')
#     # save final movie similarity
#     save_obj(np.matrix(result), 'movie-cos-similarity')

def get_partial_similarity(user_movieIds):
    user_sim = False
    for movie in user_movieIds:
        if type(user_sim) == type(False):
            user_sim = cos_sim[movieIds[str(movie)], :]
        else:
            user_sim = np.vstack([user_sim, cos_sim[movieIds[str(movie)], :]])
    return user_sim

mov_sim = movie_similarity()
#mat_mat_cos_similarity(mov_sim)
#tfidf = np.concatenate((genre_tfidf, tag_tfidf), axis=1)
norm = np.linalg.norm(mov_sim, axis=1, keepdims=True)
normalized_tfidf = mov_sim / norm
cos_sim = np.matmul(normalized_tfidf, normalized_tfidf.T)

# make id dictionarie to accelerate id access
movieIds = {}
index = 0
for movie in movies['movieId']:
    movieIds[str(movie)] = index
    index+=1 #incremrent index

def get_topN(userId, topN):
    'Returns top N recommendations'
    ratings = pd.read_csv('./data/ratings.csv')
    ratings = ratings[ratings['userId'] == userId]
    
    user_rating = np.matrix(ratings['rating']).T
    user_sim = get_partial_similarity(ratings['movieId'])
    
    # sum across colums
    sim_sum = np.sum(user_sim, axis=0)
    
    recommendation = np.divide(np.matmul(user_sim.T, user_rating), np.add(sim_sum, 1).T)

    # add movie keys row
    recommendation = np.concatenate((np.matrix(np.ones((len(recommendation),), dtype=int)*userId).T, np.matrix(list(movieIds.keys())).T.astype(int), recommendation), axis=1)
    # sort recommendations
    idx = np.lexsort((recommendation[:, 1].squeeze(), -recommendation[:, 2].squeeze()))
    recommendation = recommendation[idx].squeeze()
 
    #leave only topN terms
    recommendation = recommendation[:topN, :]
    return recommendation
