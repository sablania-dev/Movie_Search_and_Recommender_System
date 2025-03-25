import streamlit as st
import pandas as pd
from loading_and_preprocessing import load_data, preprocess_data
from functions1 import autocomplete, get_k_recommendations

# Load and display the image with a custom size
st.image("images/Netflix.jpg", width=300)  # Replace with your image filename

# Search bar with magnifying glass button
user_input = st.text_input("Search", placeholder="Search for Movies...")
search_button = st.button("🔍 Search")

# Load Real DataFrame
data = load_data(0)
data = preprocess_data(data)

# Display REAL search results when button is clicked
if search_button:
    if user_input:
        complete_input = autocomplete(data, user_input)
        filtered_df = get_k_recommendations(data, user_input, 10)
        if not filtered_df.empty:
            st.dataframe(filtered_df)
        else:
            st.write("No results found.")
    else:
        st.write("Please enter a search term.")
