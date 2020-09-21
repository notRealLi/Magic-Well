from flask import Flask, request, jsonify
import GetOldTweets3 as got
from tensorflow.keras import models
import pickle
import os
from flask_cors import CORS
from ai import sentiment
import re

app = Flask(__name__)
CORS(app)

model = models.load_model('./models/sequential.h5')
tokenizer = pickle.load(open('./models/tokenizer.pkl', 'rb'))


@app.route('/tweets/search', methods=['GET'])
def search_tweets():
    query = request.args.get('keywords')
    need_sentiment = request.args.get('sentiment')
    count = 50
    criteria = got.manager.TweetCriteria().setQuerySearch(query).setMaxTweets(count)
    tweets = got.manager.TweetManager.getTweets(criteria)
    results = [tweet.text for tweet in tweets]

    if need_sentiment != None:
        sentiment_calculator = sentiment.SentimentCalculator()
        results = [{'text': text, 'sentiment':
                    sentiment_calculator.predict(text, model, tokenizer)} for text in results]
    else:
        results = [{'text': text} for text in results]

    res = jsonify(results)

    return res


@app.route('/ai/sentiment', methods=['GET'])
def get_sentiment():
    text = request.args.get('q')
    dirty = request.args.get('dirty')

    if dirty != None:
        text = re.sub(
            '@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+', ' ', str(text).lower()).strip()

    sentiment_calculator = sentiment.SentimentCalculator()
    result = sentiment_calculator.predict(text, model, tokenizer)
    json = jsonify(result)

    return json


# Run server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8080')
