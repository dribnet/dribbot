import tweepy
import json
import argparse
import sys

parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('-m','--mode', help='mode: [text|image|images]', default="text")
args = parser.parse_args()

with open('creds.json') as data_file:    
    creds = json.load(data_file)

auth = tweepy.OAuthHandler(creds["consumer_key"], creds["consumer_secret"])
auth.set_access_token(creds["access_token"], creds["access_token_secret"])

api = tweepy.API(auth)


if args.mode == "text":
    print "text beep"
    api.update_status('beep')
elif args.mode == "image":
    print "image beep"
    api.update_with_media('images/beep.jpg')
elif args.mode == "images":
    print "multiple image beep"
    beeps = ["images/beep1.jpg", "images/beep2.jpg", "images/beep3.jpg"]
    media_ids = [api.media_upload(i).media_id_string for i in beeps]
    api.update_status(status="beep beep", media_ids=media_ids)
else:
    print "unknown mode: {}".format(args.mode)
    sys.exit(1)
