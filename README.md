# Sentiment-Analysis-Model-using-Twitter-Data
# Hate Speech Detection App

## Overview
This project is about detecting hate speech in tweets. 
With the rise of social media, hate speech has become a 
serious problem and this app helps identify racist and 
sexist tweets automatically using machine learning.

## What This App Does
You simply type or paste any tweet into the app and it 
will tell you whether the tweet contains hate speech or not. 
It also shows you the probability of the tweet being racist 
or normal which makes it more transparent and trustworthy.

## Dataset
The dataset used in this project contains around 32,000 tweets 
that are labeled as either:
- **0** → Normal Tweet
- **1** → Racist / Sexist Tweet

## How It Works
The app goes through the following steps to make a prediction:

1. **Data Cleaning** - Removes mentions, hashtags, links, 
   special characters and converts text to lowercase

2. **Stemming** - Reduces words to their root form 
   for example "running" becomes "run"

3. **Bag of Words** - Converts cleaned text into 
   numbers that the machine learning model can understand

4. **Logistic Regression** - A simple but powerful machine 
   learning model that classifies the tweet as hate speech or not

## Technologies Used
- Python
- Streamlit
- Scikit-learn
- NLTK
- Pandas
- NumPy
- Pickle

## How To Run Locally

### Step 1 - Clone the repository
```bash
git clone https://github.com/yourusername/hate-speech-detection.git
cd hate-speech-detection
```

### Step 2 - Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 - Run the app
```bash
streamlit run app.py
```

## Project Structure
```
hate-speech-detection/
│
├── app.py               # Main Streamlit application
├── model.pkl            # Trained Logistic Regression model
├── bow.pkl              # Bag of Words vectorizer
├── requirements.txt     # Required libraries
└── README.md            # Project documentation
```

## Sample Predictions

| Tweet | Prediction |
|-------|-----------|
| happy birthday to my best friend | Normal |
| women should not be allowed to work | Racist/Sexist |
| just had the best coffee this morning | Normal |
| these people are destroying our country | Racist/Sexist |

## Model Performance
- **Algorithm:** Logistic Regression
- **Accuracy:** ~95%
- **Feature Extraction:** Bag of Words (top 1000 features)
- **Train/Test Split:** 80% training, 20% testing

## Deployment
This app is deployed on Streamlit Cloud.
You can access it here:https://share.streamlit.io

## Author
This project was developed as part of a Natural Language 
Processing (NLP) task to classify hate speech in social 
media posts. The goal was to build a simple yet effective 
pipeline from raw text to a deployable machine learning app.

## Note
This app is built for educational purposes only. The model 
is not perfect and may sometimes misclassify tweets. Always 
use human judgment alongside any automated tool.
