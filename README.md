# HistoryBot: An NLP Chatbot for Historical Event Exploration

**Author:** Pablo Gonzalez  
**Course:** MSDS 453 – Natural Language Processing  
**Quarter:** Spring 2025  
**Date:** 06/08/2025

---

## Project Overview

**HistoryBot** is a domain-specific chatbot designed to make learning about historical events more interactive and accessible. The chatbot understands natural language queries related to dates, historical events, and key figures, and retrieves relevant information using machine learning and NLP techniques.

---

## Dataset

- **Source:** [Kaggle – World Important Events: Ancient to Modern](https://www.kaggle.com/datasets/saketk511/world-important-events-ancient-to-modern)
- **Description:** This dataset contains over 1,000 historical events with metadata including event name, description, date, location, and key people involved.

---

## Methodology

The chatbot was developed using the standard NLP pipeline:

### 1. Data Cleaning and Preprocessing
- No missing values in the original dataset.
- Applied lowercasing, punctuation and whitespace removal, and date standardization using Python (`re`, `string`, `pandas`).

### 2. Intent Classification
- Used `TF-IDF` vectorization on synthetic user queries.
- Trained a `Logistic Regression` classifier to categorize questions into:
  - **Date-related**
  - **Event-related**
  - **Person-related**
- Synthetic training data was generated using **ChatGPT** to address the limited availability of labeled samples.

### 3. Model Tuning
- Applied `GridSearchCV` to tune hyperparameters: `C`, `penalty`, `class_weight`, and `ngram_range`.
- Final parameters: 
  - `C = 1.0`
  - `penalty = l2`
  - `solver = liblinear`
  - `ngram_range = (1, 3)`

### 4. Event Retrieval
- Separate TF-IDF vectorizer trained on historical event descriptions.
- Used cosine similarity to return the most relevant historical entry based on the query intent.

### 5. Deployment
- Built using [**Streamlit**](https://streamlit.io) for an interactive interface.
- `joblib` used to save trained models; files were hosted on GitHub.
- Users can input any free-text query and receive an intelligent, context-aware historical answer.

---

## Results Summary

| Class  | Precision | Recall | F1 Score |
|--------|-----------|--------|----------|
| Date   | 1.00      | 0.50   | 0.67     |
| Event  | 0.62      | 1.00   | 0.77     |
| Person | 1.00      | 0.50   | 0.67     |
| **Accuracy** |        |        | **0.73**  |
| **Macro Avg** | 0.88 | 0.67   | 0.70     |
| **Weighted Avg** | 0.83 | 0.73 | 0.71     |

---

## References

OpenAI. (2024). ChatGPT. Retrieved from https://openai.com/chatgpt

Saket Kumar. (2023). World Important Events – Ancient to Modern [Dataset]. Kaggle. Retrieved from https://www.kaggle.com/datasets/saketk511/world-important-events-ancient-to-modern

Streamlit Inc. (2024). Streamlit. Retrieved from https://streamlit.io
