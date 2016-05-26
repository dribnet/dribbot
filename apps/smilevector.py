import tweepy
import json
import argparse
import sys
import re
import urllib
import urlparse, os
import doalign
from subprocess import call

def do_convert(infile, outfile):
    return doalign.align_face(local_media, final_media, 128)

#
# TODO: this logic could be refactored from sample and put into library
#
# Example of how to track an account and repost images with modifications.
# 
# stores last tweet id temp_files subdirectory
# (also depends on imagemagick)
#

parser = argparse.ArgumentParser(description='Follow account and repost munged images')
parser.add_argument('-a','--account', help='Account to follow', default="people")
parser.add_argument('-d','--debug', help='Debug: do not post', default=False, action='store_true')
parser.add_argument('-o','--open', help='Open image (when in debug mode)', default=False, action='store_true')
parser.add_argument('-n','--no-update', dest='no_update',
        help='Do not update postion on timeline', default=False, action='store_true')

args = parser.parse_args()

with open('creds.json') as data_file:    
    creds = json.load(data_file)

tempfile = "temp_files/{}_follow_account_lastid.txt".format(args.account)

# try to recover last known tweet_id (if fails, last_id will stay None)
last_id = None
try:
	f = open(tempfile).read()
	last_id = int(f)
except IOError: 
	pass

auth = tweepy.OAuthHandler(creds["consumer_key"], creds["consumer_secret"])
auth.set_access_token(creds["access_token"], creds["access_token_secret"])

api = tweepy.API(auth)

if last_id is None:
    # just grab most recent tweet
    stuff = api.user_timeline(screen_name = args.account, \
        count = 1, \
        include_rts = True)
else:
    # look back up to 100 tweets since last one and then show next one
    stuff = api.user_timeline(screen_name = args.account, \
        count = 100, \
        since_id = last_id,
        include_rts = True)

if len(stuff) == 0:
    print("(nothing to do)")
    sys.exit(0)

top = stuff[-1]._json
tweet_id = top["id"]
rawtext = top["text"]
text = re.sub(' http.*$', '', rawtext)
media = top["entities"]["media"][0]
media_url = media["media_url"]
link_url = u"https://twitter.com/{}/status/{}".format(args.account, tweet_id)

path = urlparse.urlparse(media_url).path
ext = os.path.splitext(path)[1]
local_media = "temp_files/media_file{}".format(ext)
final_media = "temp_files/final_file{}".format(ext)

urllib.urlretrieve(media_url, local_media)

result = do_convert(local_media, final_media)

media_id = api.media_upload(final_media).media_id_string
update_text = u"{}\n{}".format(text, link_url)
if args.debug:
    print(u"Update text: {}, Image: {}".format(update_text, final_media))
    if not result:
        print("(Image conversation failed)")
        if args.open:
            call(["open", local_media])
    else:
        if args.open:
            call(["open", final_media])
else:
    if result:
        api.update_status(status=update_text, media_ids=[media_id])
        print(u"Posted: {}".format(update_text))
    else:
        print(u"Skipped: {}".format(update_text))

# success, update last known tweet_id
if not args.no_update:
    with open(tempfile, 'w') as f:
      f.write("{}".format(tweet_id))
