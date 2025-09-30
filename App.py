import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity



# ----------------------------
# Load Data
# ----------------------------
ratings = pd.read_csv("Ratings.csv", encoding="latin-1")
books = pd.read_csv("Books.csv", encoding="latin-1")
users = pd.read_csv("Users.csv", encoding="latin-1")

# ----------------------------
# Merge & Basic Cleaning
# ----------------------------
book_rating = pd.merge(ratings, books, on='ISBN')
data = pd.merge(book_rating, users, on='User-ID')

# Drop unused image columns
data.drop(['Image-URL-L', 'Image-URL-M', 'Image-URL-S'], axis=1, inplace=True)

# Fill NaN ages with average
avg_age = np.round(data['Age'].mean(), 0)
data['Age'].fillna(avg_age, inplace=True)

# Handle outliers in Age
Q1, Q3 = data['Age'].quantile([0.25, 0.75])
IQR = Q3 - Q1
l_bound, u_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
med_age = data['Age'].median()

data['Age'] = np.where(
    (data['Age'] < l_bound) | (data['Age'] > u_bound),
    med_age,
    data['Age']
)

# ----------------------------
# Content-Based Recommendation
# ----------------------------
tfidf = TfidfVectorizer(stop_words="english")

# Choose text column for content-based similarity
if "description" in books.columns:
    content_data = books["description"].fillna("")
elif "genres" in books.columns:
    content_data = books["genres"].fillna("")
else:
    content_data = books["Book-Title"].fillna("")  # Many Book datasets use "Book-Title"

tfidf_matrix = tfidf.fit_transform(content_data)

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix,dense_output=False)

books = books.reset_index(drop=True)

# ----------------------------
# Recommendation Function
# ----------------------------
def recommend(title, top_n=5):
    if title not in books['Book-Title'].values:
        return ["Book not found! Please try another title."]

    idx = books[books['Book-Title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # skip itself
    book_indices = [i[0] for i in sim_scores]
    return books['Book-Title'].iloc[book_indices].tolist()

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Book Recommender", page_icon="📚", layout="centered")

st.title("📚 Book Recommendation System")
st.write("Select a book and get similar recommendations using content-based filtering.")

selected_book = st.selectbox("Choose a book:", books['Book-Title'].values)

if st.button("Recommend Similar Books"):
    recommendations = recommend(selected_book, top_n=5)
    st.subheader("📖 Recommended Books:")
    for rec in recommendations:
        st.write(f"- {rec}")
