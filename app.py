from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
import GetOldTweets3 as got
import os

from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/tweets/search', methods=['GET'])
def search_tweets():
    query = request.args.get('keywords')
    count = 100
    criteria = got.manager.TweetCriteria().setQuerySearch(query).setMaxTweets(count)
    tweets = got.manager.TweetManager.getTweets(criteria)
    results = [tweet.text for tweet in tweets]
    res = jsonify(results)

    return res


# Run server
if __name__ == '__main__':
    app.run()
