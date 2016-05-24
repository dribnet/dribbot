import tweepy
import json

#
# Small "hello world" tweepy program shows how to use API and
# prints out public tweets from timeline.
#

with open('creds.json') as data_file:    
    creds = json.load(data_file)

auth = tweepy.OAuthHandler(creds["consumer_key"], creds["consumer_secret"])
auth.set_access_token(creds["access_token"], creds["access_token_secret"])

api = tweepy.API(auth)

public_tweets = api.home_timeline()
for tweet in public_tweets:
    print tweet.text