import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
faq = pd.read_csv("hr_faq_500.csv")

# Prepare TF-IDF model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(faq["Question"])

print("🤖 HR Chatbot is online! (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Chatbot: Goodbye! 👋")
        break

    # Transform user query
    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, X).flatten()
    idx = similarities.argmax()
    score = similarities[idx]

    # Threshold to avoid wrong matches
    if score > 0.3:  
        print("Chatbot:", faq.iloc[idx]["Answer"])
    else:
        print("Chatbot: Sorry, I don’t know the answer to that.")
