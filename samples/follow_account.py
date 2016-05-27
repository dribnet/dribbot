import tweepy
import json
import argparse
import sys
import re
import urllib
import urlparse, os
from subprocess import call

#
# Example of how to track an account and repost images with modifications.
# 
# stores last tweet id temp_files subdirectory
# (also depends on imagemagick)
#

if __name__ == "__main__":
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
    local_media = "media_file{}".format(ext)
    final_media = "final_file{}".format(ext)

    urllib.urlretrieve(media_url, local_media)

    call(["convert", "-negate", local_media, final_media])

    media_id = api.media_upload(final_media).media_id_string
    if args.debug:
        print("Update text: {}, Image: {}".format(text, final_media))    
        if args.open:
            call(["open", final_media])
    else:
        status = api.update_status(status=text, media_ids=[media_id])
        # print(status.id, status.user)
        posted_id = status.id
        posted_name = status.user.name
        print("Posted: {} ({} -> {})".format(text, posted_name, posted_id))
        respond_text = u"@{} {}".format(posted_name, link_url)
        api.update_status(status=respond_text, in_reply_to_status_id=posted_id)

        # success, update last known tweet_id
    if not args.no_update:
        with open(tempfile, 'w') as f:
          f.write("{}".format(tweet_id))
