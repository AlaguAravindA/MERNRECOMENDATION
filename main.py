from flask import Flask, jsonify, request
from flask_cors import CORS
import tensorflow as tf
from fuzzywuzzy import fuzz
import pandas as pd

# Load the compressed model file
loaded_model = tf.keras.models.load_model('path_to_your/model.h5')
# Load the pretrained model
model = loaded_model

# Load the movie data (assuming you have it stored in a DataFrame)
movie_data = pd.read_csv('movies_list.csv')

# Create a dictionary mapping lowercase movie titles to indices
movie_title_index_map = {title.lower(): index for index, title in enumerate(movie_data["original_title"])}

app = Flask(__name__)
CORS(app)

# Replace this with your recommendation function or model
def get_movie_recommendations(input_title, num_recommendations=5):
    input_title_lower = input_title.lower()

    # Check if the input title exists in the map
    if input_title_lower not in movie_title_index_map:
        return []

    index = movie_title_index_map[input_title_lower]

    # Get recommendations using the pretrained model
    recommendations_indices = model.predict([[index]])  # Assuming the model expects movie indices as input

    # Sort the recommendation indices based on their scores
    sorted_indices = recommendations_indices.argsort()[0]

    # Convert recommendation indices to movie titles
    recommendations = []
    for idx in sorted_indices[:num_recommendations]:
        if idx < len(movie_data):
            movie_title = movie_data.loc[int(idx), "original_title"]
            recommendations.append(movie_title)

    # Return top num_recommendations recommended movie titles
    return recommendations

@app.route('/recommendations', methods=['GET'])
def recommendations():
    # Get the movie title from the query parameter
    movie_title = request.args.get('movie_title')

    if not movie_title:
        return jsonify({"error": "Movie Title is required"}), 400

    # Call your recommendation function or model
    recommended_movies = get_movie_recommendations(movie_title)

    # Return the recommended movies as JSON
    return jsonify({"recommendations": recommended_movies})

if __name__ == '__main__':
    app.run()
