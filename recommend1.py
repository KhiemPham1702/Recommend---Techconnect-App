import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
# Load data
# products = pd.read_csv('./data/Product.csv')
# ratings = pd.read_csv('./data/Rating_pro.csv')
with open('./data/test_pro.json',  'rb') as f:
    products = pd.json_normalize(json.load(f))
with open('./data/test_rating.json',  'rb') as f:
    ratings = pd.json_normalize(json.load(f))

# try:
#     with open('./data/test_pro.json',  'rb') as f:
#         products = pd.json_normalize(json.load(f))
#     with open('./data/test_rating.json',  'rb') as f:
#         ratings = pd.json_normalize(json.load(f))
# except Exception as e:
#     with open('./data/Product.json',  'rb') as f:
#         products = pd.json_normalize(json.load(f))
#     with open('./data/Rating_pro.json',  'rb') as f:
#         ratings = pd.json_normalize(json.load(f))

# Merge dataframes
df = pd.merge(products, ratings, on='Id_product')

# Compute the mean rating for each product
mean_ratings = df.groupby(['Name_product'])['Rating'].mean().reset_index()

# Create a TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Replace NaN with an empty string
products['Brand'] = products['Brand'].fillna('')

# Construct the TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(products['Brand'])

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Define a function to get content-based recommendations
def get_content_based_recommendations(title, cosine_sim=cosine_sim, products=products):
    # Get the index of the product that matches the title
    idx = products[products['Name_product'] == title].index[0]

    # Get the similarity scores of all products with that product
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the products based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top 10 most similar products
    sim_scores = sim_scores[1:11]

    # Get the product indices
    product_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar products
    return products['Name_product'].iloc[product_indices]

# Define a function to get collaborative filtering recommendations
def get_collaborative_filtering_recommendations(user_id, mean_ratings=mean_ratings, ratings=ratings):
    # Get the products that the user has rated
    user_products = ratings[ratings['Id_user'] == user_id]['Id_product']

    # Get the mean rating for each product that the user has rated
    user_mean_ratings = mean_ratings[mean_ratings['Name_product'].isin(user_products)]

    # Sort the products based on the mean rating
    user_mean_ratings = user_mean_ratings.sort_values(by='Rating', ascending=False)

    # Get the top 10 highest rated products
    user_mean_ratings = user_mean_ratings.head(10)

    # Return the top 10 highest rated products
    return user_mean_ratings['Name_product']

# Define a function to get hybrid recommendations
def get_hybrid_recommendations(user_id, title, cosine_sim=cosine_sim, products=products, mean_ratings=mean_ratings, ratings=ratings):
    # Get the content-based recommendations
    content_based_recommendations = get_content_based_recommendations(title)

    # Get the collaborative filtering recommendations
    collaborative_filtering_recommendations = get_collaborative_filtering_recommendations(user_id)

    # Merge the recommendations
    recommendations = pd.concat([content_based_recommendations, collaborative_filtering_recommendations]).drop_duplicates().reset_index(drop=True)

    # Return the recommendations
    return recommendations

# Get hybrid recommendations for user 1 and the product
