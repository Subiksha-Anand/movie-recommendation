# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 15:34:25 2025

@author: subik
"""

import numpy as np
import pandas as pd
import difflib
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies_data = pd.read_csv("movies.csv")

# Ensure 'index' column exists
if 'index' not in movies_data.columns:
    movies_data.reset_index(inplace=True)

# Select important features
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
movies_data[selected_features] = movies_data[selected_features].fillna('')

# Combine selected features into a single text representation
combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

# Convert text data into TF-IDF feature vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Compute cosine similarity
similarity = cosine_similarity(feature_vectors)

def recommend_movies(movie_name):
    """Returns a list of recommended movies based on the input movie."""
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    
    if not find_close_match:
        return ["No close matches found. Please try another movie."]
    
    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data.title == close_match].index[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    
    recommendations = [movies_data.iloc[movie[0]]['title'] for movie in sorted_similar_movies[1:11]]
    return recommendations

def main():
    """Streamlit UI for the movie recommendation system."""
    st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="centered")
    st.markdown("""
        <style>
            .main {background-color: #f0f2f6;}
            .stButton>button {background-color: #4CAF50; color: white; font-size: 16px; padding: 10px;}
            .stTextInput>div>div>input {background-color: #ffffff; border-radius: 10px;}
        </style>
    """, unsafe_allow_html=True)
    
    st.title('🎬 Movie Recommendation System')
    st.markdown("<h3 style='color: #4CAF50;'>Find movies similar to your favorite!</h3>", unsafe_allow_html=True)
    
    movie_name = st.text_input("Enter your favorite movie name:", placeholder="Type here...")
    
    recommendations = []
    
    if st.button('🎥 Get Recommendations'):
        recommendations = recommend_movies(movie_name)
    
    if recommendations:
        st.subheader("🎯 Movies Suggested for You:")
        st.markdown("<ul style='color: #333; font-size: 18px;'>", unsafe_allow_html=True)
        for i, movie in enumerate(recommendations, start=1):
            st.markdown(f"<li><b>{i}. {movie}</b></li>", unsafe_allow_html=True)
        st.markdown("</ul>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
