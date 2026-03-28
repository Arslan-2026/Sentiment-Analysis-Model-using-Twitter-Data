import nltk
import os
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
import streamlit as st
import pickle
import re
from nltk.stem import PorterStemmer
model = pickle.load(open('model.pkl', 'rb'))
bow = pickle.load(open('bow.pkl', 'rb'))
ps = PorterStemmer()

def clean_tweet(text):
    text = str(text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = text.split()
    stemmed = [ps.stem(word) for word in tokens]
    return ' '.join(stemmed)


st.title("Hate Speech Detection App")
st.subheader("Detect Racist/Sexist Tweets")


tweet = st.text_area("Enter Tweet Here:")

if st.button("Predict"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet!")
    else:
        cleaned = clean_tweet(tweet)
        vectorized = bow.transform([cleaned])
        
        prediction = model.predict(vectorized)
        probability = model.predict_proba(vectorized)
        
        if prediction[0] == 1:
            st.error(f"Racist/Sexist Tweet Detected!")
        else:
            st.success(f"Normal Tweet")
            
        st.write(f"**Probability Normal:** {probability[0][0]:.2f}")
        st.write(f"**Probability Racist:** {probability[0][1]:.2f}")
