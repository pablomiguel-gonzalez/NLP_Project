import streamlit as st
import joblib
import pandas as pd
import re
import string
from sklearn.metrics.pairwise import cosine_similarity

# Load your models
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
intent_classifier = joblib.load('intent_classifier.pkl')

# Load your data
world_events = pd.read_csv('World Important Dates.csv')

# Define your lightweight preprocessing function
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        text = re.sub(r'\s+', ' ', text).strip()
    return text

world_events['Name_of_Incident_clean'] = world_events['Name of Incident'].apply(clean_text)

# Define your search function (reuse your TF-IDF search code here too)
from sklearn.metrics.pairwise import cosine_similarity

def search_event(query, top_n=3):
    query_clean = clean_text(query)
    query_vec = tfidf_vectorizer.transform([query_clean])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    return world_events.iloc[top_indices]

# IMPORTANT: You also need to precompute the TF-IDF matrix of all event names
tfidf_matrix = tfidf_vectorizer.transform(world_events['Name_of_Incident_clean'])

# Example start
st.title("📚 HistoryBot: Your Historical Events Assistant")
st.write("Ask me about historical events, dates, and important figures!")

user_input = st.text_input("Ask your question about history:")

if user_input:
    # Intent prediction
    intent = intent_classifier.predict([query_clean])[0]

    
    # TF-IDF Retrieval
    results = search_event(user_input, top_n=1)
    
    if not results.empty:
        event_name = results.iloc[0]['Name_of_Incident']
        impact = results.iloc[0]['Impact']
        year = results.iloc[0]['Year_Clean']
        country = results.iloc[0]['Country']
        place = results.iloc[0]['Place_Name']
        
        if year < 0:
            year_str = f"{abs(year)} BC"
        else:
            year_str = str(year)
        
        # Response based on intent
        if intent == "date":
            response = f"🗓️ The event '{event_name}' happened in {year_str}."
        elif intent == "person":
            response = f"🧑‍🎓 '{event_name}' involved notable figures. Would you like to know more about the key people involved?"
        else:
            response = f"📜 {event_name} ({year_str})\nImpact: {impact}\nLocation: {place.title()}, {country.title()}."
        
        st.success(response)
    else:
        st.error("Sorry, I couldn't find any information about that.")
