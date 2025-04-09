import streamlit as st
import pickle
import polars as pl
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

# App title and description
st.title("📚 Book Recommendation System")
st.markdown("Enter a book summary and genres to get personalized book recommendations!")

# Load the TF-IDF vectorizer
@st.cache_resource
def load_models():
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    
    # Load the KNN model
    with open('knn_model.pkl', 'rb') as f:
        knn_model = pickle.load(f)
    
    return tfidf, knn_model

# Load the dataset
@st.cache_data
def load_data():
    df_lazy = pl.scan_csv('goodreadsV2.csv')
    df_cleaned = (
        df_lazy.drop_nulls(subset=['name', 'summary', 'genres'])
        .with_columns([
            (pl.col('summary') + ' ' + pl.col('genres')).alias('combined_features')
        ])
    ).collect()
    
    # Apply preprocessing to create the 'processed_features' column
    df_cleaned = df_cleaned.with_columns([
        pl.col('combined_features')
        .map_elements(preprocess_text, return_dtype=pl.Utf8)
        .alias('processed_features')
    ])
    
    return df_cleaned

# Define the preprocessing function
def preprocess_text(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())

# Recommendation function for out-of-dataset books
def recommend_books_knn_out_of_dataset(input_summary, input_genres, top_n=5):
    # Combine and preprocess the input book's features
    combined_input = f"{input_summary} {input_genres}"
    processed_input = preprocess_text(combined_input)

    # Transform the input book's features using the loaded TF-IDF vectorizer
    input_vector = tfidf.transform([processed_input])

    # Find the nearest neighbors using the loaded KNN model
    distances, indices = knn_model.kneighbors(input_vector, n_neighbors=top_n)

    # Retrieve the recommended book titles and additional information
    recommendations = []
    for i, idx in enumerate(indices.flatten()):
        book_info = {
            "title": df_cleaned['name'][idx],
            "summary": df_cleaned['summary'][idx],
            "genres": df_cleaned['genres'][idx],
            "similarity_score": 1 - distances.flatten()[i]  # Convert distance to similarity
        }
        recommendations.append(book_info)

    return recommendations

# Load models and data
try:
    tfidf, knn_model = load_models()
    df_cleaned = load_data()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models or data: {e}")
    models_loaded = False

# Sidebar for inputs
st.sidebar.header("Input Parameters")

# Input fields
input_summary = st.sidebar.text_area("Book Summary", 
                                    placeholder="Enter a brief summary of the book...",
                                    height=150)

input_genres = st.sidebar.text_input("Genres", 
                                    placeholder="E.g., fantasy, adventure, mystery")

# Number of recommendations slider
num_recommendations = st.sidebar.slider("Number of Recommendations", 
                                        min_value=1, 
                                        max_value=10, 
                                        value=5)

# Get recommendations button
if st.sidebar.button("Get Recommendations") and models_loaded:
    if input_summary and input_genres:
        with st.spinner("Finding the perfect books for you..."):
            # Get recommendations
            recommendations = recommend_books_knn_out_of_dataset(
                input_summary, 
                input_genres, 
                top_n=num_recommendations
            )
            
            # Display recommendations
            st.header("Recommended Books")
            
            # Create columns for book cards
            cols = st.columns(min(3, num_recommendations))
            
            for i, book in enumerate(recommendations):
                col_idx = i % 3
                with cols[col_idx]:
                    st.subheader(book["title"])
                    st.markdown(f"**Genres:** {book['genres']}")
                    st.markdown(f"**Similarity Score:** {book['similarity_score']:.2f}")
                    with st.expander("Summary"):
                        st.write(book["summary"])
                    st.divider()
            
            # Visualization of similarity scores
            st.header("Similarity Scores")
            fig, ax = plt.subplots(figsize=(10, 5))
            
            book_titles = [book["title"] for book in recommendations]
            similarity_scores = [book["similarity_score"] for book in recommendations]
            
            # Create horizontal bar chart
            sns.barplot(x=similarity_scores, y=book_titles, palette="viridis", ax=ax)
            ax.set_xlabel("Similarity Score")
            ax.set_ylabel("Book Title")
            ax.set_title("Book Recommendation Similarity Scores")
            
            st.pyplot(fig)
            
    else:
        st.warning("Please enter both a summary and genres to get recommendations.")

# Add some information about the app
st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.info(
    """
    This app uses TF-IDF vectorization and K-Nearest Neighbors to recommend books 
    based on your input summary and genres.
    
    The recommendations are based on textual similarity between your input and 
    our database of books from Goodreads.
    """
)

# Add a footer
st.markdown("---")
st.markdown("📚 Book Recommendation System | Created with Streamlit")
