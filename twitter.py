import tweepy
import schedule
import time
from sklearn.ensemble import RandomForestRegressor
from transformers import pipeline
from textblob import TextBlob
import numpy as np

# Twitter Authentication
auth = tweepy.OAuthHandler("API_KEY", "API_SECRET_KEY")
auth.set_access_token("ACCESS_TOKEN", "ACCESS_TOKEN_SECRET")
api = tweepy.API(auth)

# 1. Data Collection
def fetch_trending_tweets(keyword, count=100):
    tweets = api.search_tweets(q=keyword, lang='en', result_type='popular', count=count)
    return tweets

def extract_features(tweet):
    text = tweet.text
    sentiment = TextBlob(text).sentiment.polarity
    length = len(text)
    return np.array([sentiment, length])

# 2. ML Model Setup
def train_model(tweets):
    X = np.array([extract_features(tweet) for tweet in tweets])
    y = np.array([tweet.favorite_count + tweet.retweet_count for tweet in tweets])
    
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    return model

# 3. Content Generation & Optimization
def paraphrase_text(text):
    paraphraser = pipeline("text2text-generation", model="t5-base")
    return paraphraser(text, max_length=50, do_sample=False)[0]['generated_text']

def generate_and_post_tweet(model, tweets):
    best_tweet = max(tweets, key=lambda tweet: model.predict([extract_features(tweet)]))
    new_content = paraphrase_text(best_tweet.text)
    
    sentiment = TextBlob(new_content).sentiment.polarity
    if sentiment > 0:  # Ensure positive sentiment
        api.update_status(new_content)

# 4. Automation & Scheduling
def run_daily(keyword):
    tweets = fetch_trending_tweets(keyword)
    model = train_model(tweets)
    generate_and_post_tweet(model, tweets)

# Schedule the function to run at specific times
schedule.every().day.at("10:00").do(run_daily, keyword="technology")
schedule.every().day.at("14:00").do(run_daily, keyword="technology")
schedule.every().day.at("18:00").do(run_daily, keyword="technology")

# 5. Continuous Loop
while True:
    schedule.run_pending()
    time.sleep(1)
