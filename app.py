import streamlit as st
import joblib
import pandas as pd

# Load Models
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
intent_classifier = joblib.load('intent_classifier.pkl')

# Load Dataset
world_events = pd.read_csv('World Important Dates.csv')
# Your HistoryBot code here...

# Example start
st.title("📚 HistoryBot: Your Historical Events Assistant")
st.write("Ask me about historical events, dates, and important figures!")

user_input = st.text_input("Ask your question about history:")

if user_input:
    # Intent prediction
    query_clean = clean_text(user_input)
    query_vec = intent_vectorizer.transform([query_clean])
    intent = intent_classifier.predict(query_vec)[0]
    
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
