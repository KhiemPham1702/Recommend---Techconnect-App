import pandas as pd
from surprise import Dataset, Reader, SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np
# Load data
# products = pd.read_csv('./data/Product.csv')
# ratings = pd.read_csv('./data/Rating_pro.csv')
# product_genres = pd.read_csv('./data/Product.csv')

with open('./data/test_pro.json',  'rb') as f:
    products = pd.json_normalize(json.load(f))
with open('./data/test_rating.json',  'rb') as f:
    ratings = pd.json_normalize(json.load(f))
with open('./data/test_pro.json',  'rb') as f:
    product_genres = pd.json_normalize(json.load(f))

# try:
#     with open('./data/test_pro.json',  'rb') as f:
#         products = pd.json_normalize(json.load(f))
#     with open('./data/test_rating.json',  'rb') as f:
#         ratings = pd.json_normalize(json.load(f))
#     with open('./data/test_pro.json',  'rb') as f:
#         product_genres = pd.json_normalize(json.load(f))
# except Exception as e:
#     with open('./data/Product.json',  'rb') as f:
#         products = pd.json_normalize(json.load(f))
#     with open('./data/Rating_pro.json',  'rb') as f:
#         ratings = pd.json_normalize(json.load(f))
#     with open('./data/Product.json',  'rb') as f:
#         product_genres = pd.json_normalize(json.load(f))

# Define a function to get collaborative filtering recommendations
def get_collaborative_filtering_recommendations(user_id, products=products, ratings=ratings):
    # Define the rating scale
    reader = Reader(rating_scale=(0.5, 5))

    # Load the data into the Surprise dataset format
    data = Dataset.load_from_df(ratings[['Id_user', 'Id_product', 'Rating']], reader)

    # Build the similarity matrix
    algo = SVD()
    trainset = data.build_full_trainset()
    algo.fit(trainset)

    # Get the products that the user has rated
    user_products= ratings[ratings['Id_user'] == user_id]['Id_product']

    # Get the products that the user has not rated
    other_products = set(products['Id_product']) - set(user_products)

    # Predict the ratings for the products that the user has not rated
    testset = [(user_id, movie_id, 0) for movie_id in other_products]
    predictions = algo.test(testset)

    # Get the top N recommendations based on predicted ratings
    top_n = sorted(predictions, key=lambda x: x.est, reverse=True)[:5]
    top_n = [int(x.iid) if not np.isnan(x.iid) else 1 for x in top_n]

    return top_n

# Define a function to get content-based filtering recommendations
def get_content_based_filtering_recommendations(user_id, products=products, product_genres=product_genres):
    # Get the products that the user has rated
    user_products = ratings[ratings['Id_user'] == user_id]['Id_product']

    # Get the products genres for the user's rated products
    user_product_genres = product_genres[product_genres['Id_product'].isin(user_products)]['Brand'].tolist()

    # Vectorize the products genres
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(user_product_genres)

    # Calculate the cosine similarity between the user's products genres and all other products
    cosine_similarities = cosine_similarity(X, vectorizer.transform(product_genres['Brand']))

    # Get the top N recommendations based on cosine similarities
    similar_indices = cosine_similarities.argsort()[0][-5:]
    top_n = [int(product_genres.iloc[i]['Id_product']) if not pd.isna(product_genres.iloc[i]['Id_product']) else 1
             for i in similar_indices]

    return top_n

# Define a function to get hybrid recommendations
def get_hybrid_recommendations(user_id, products=products, ratings=ratings, product_genres=product_genres):
    # Get the collaborative filtering recommendations
    cf_recommendations = get_collaborative_filtering_recommendations(user_id, products, ratings)

    # Get the content-based filtering recommendations
    cb_recommendations = get_content_based_filtering_recommendations(user_id, products, product_genres)

    # Combine the two sets of recommendations
    hybrid_recommendations = list(set(cf_recommendations + cb_recommendations))

    return hybrid_recommendations

def get_data_hybrid(user_id):
    return get_hybrid_recommendations(user_id, products, ratings, product_genres)
# Get hybrid recommendations for user 1


