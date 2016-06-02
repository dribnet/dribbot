import tweepy
import json
import argparse
import sys
import re
import urllib
import urlparse, os
from shutil import copyfile
import doalign
from subprocess import call

image_size = 128

# discgen related imports
from blocks.model import Model
from blocks.serialization import load
from blocks.select import Selector
from utils.sample_utils import offset_from_string, anchors_from_image, get_image_vectors, compute_splash, get_json_vectors
from utils.sample import grid_from_latents
import faceswap

def do_convert(infile, outfile1, outfile2, model, smile_offset):
    aligned_file = "temp_files/aligned_file.png"
    smile1_file = "temp_files/smile1_file.png"
    smile2_file = "temp_files/smile2_file.png"

    # first try to align the face
    if not doalign.align_face(local_media, aligned_file, image_size):
        return False

    # now try to force a smile

    # first encode image to vector
    _, _, anchor_images = anchors_from_image(aligned_file, image_size=(image_size, image_size))
    anchors = get_image_vectors(model, anchor_images)

    # fire up decoder
    selector = Selector(model.top_bricks)
    decoder_mlp, = selector.select('/decoder_mlp').bricks
    z_dim = decoder_mlp.input_dim

    # TODO: this is overkill
    z = compute_splash(rows=1, cols=1, dim=z_dim, space=1, anchors=anchors, spherical=True, gaussian=True)
    z1 = z + smile_offset
    grid_from_latents(z1, model, rows=1, cols=1, anchor_images=anchor_images, tight=True, shoulders=False, save_path=smile1_file)

    z2 = z - smile_offset
    grid_from_latents(z2, model, rows=1, cols=1, anchor_images=anchor_images, tight=True, shoulders=False, save_path=smile2_file)

    try:
        faceswap.do_faceswap(infile, smile1_file, outfile1)
        faceswap.do_faceswap(infile, smile2_file, outfile2)
    except faceswap.NoFaces:
        print("faceswap: no faces in {}".format(infile))
        return False
    except faceswap.TooManyFaces:
        print("faceswap: too many faces in {}".format(infile))
        return False

    return True

#
# TODO: this logic could be refactored from sample and put into library
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
    parser.add_argument('-s','--single', help='Process only a single image', default=False, action='store_true')
    parser.add_argument('-1','--creds1', help='Twitter json credentials1 (smile)', default='forcedsmilebot.json')
    parser.add_argument('-2','--creds2', help='Twitter json credentials2 (antismile)', default='wipedsmilebot.json')
    parser.add_argument('-n','--no-update', dest='no_update',
            help='Do not update postion on timeline', default=False, action='store_true')
    parser.add_argument("--model", dest='model', type=str, default=None,
                        help="path to the saved model")
    parser.add_argument('--anchor-offset', dest='anchor_offset', default=None,
                        help="use json file as source of each anchors offsets")

    args = parser.parse_args()

    # initialize and then lazily load
    model = None
    smile_offset = None

    # now fire up tweepy
    with open(args.creds1) as data_file:
        creds1 = json.load(data_file)
    with open(args.creds2) as data_file:
        creds2 = json.load(data_file)

    tempfile = "temp_files/{}_follow_account_lastid.txt".format(args.account)

    # try to recover last known tweet_id (if fails, last_id will stay None)
    last_id = None
    try:
        f = open(tempfile).read()
        last_id = int(f)
    except IOError:
        pass

    auth1 = tweepy.OAuthHandler(creds1["consumer_key"], creds1["consumer_secret"])
    auth1.set_access_token(creds1["access_token"], creds1["access_token_secret"])
    api1 = tweepy.API(auth1)

    auth2 = tweepy.OAuthHandler(creds2["consumer_key"], creds2["consumer_secret"])
    auth2.set_access_token(creds2["access_token"], creds2["access_token_secret"])
    api2 = tweepy.API(auth2)

    if last_id is None:
        # just grab most recent tweet
        stuff = api1.user_timeline(screen_name = args.account, \
            count = 1, \
            include_rts = False,
            exclude_replies = False)
    else:
        # look back up to 100 tweets since last one and then show next one
        stuff = api1.user_timeline(screen_name = args.account, \
            count = 100, \
            since_id = last_id,
            include_rts = False,
            exclude_replies = False)

    if len(stuff) == 0:
        print("(nothing to do)")
        sys.exit(0)

    if args.single:
        stuff = [ stuff[-1] ]

    have_posted = False

    # success, update last known tweet_id
    if not args.no_update and len(stuff) > 0:
        last_item = stuff[0]._json
        last_tweet_id = last_item["id"]
        if os.path.isfile(tempfile):
            copyfile(tempfile, "{}.bak".format(tempfile))
        with open(tempfile, 'w') as f:
          f.write("{}\n".format(last_tweet_id))

    # for item in reversed(stuff):
    cur_stuff = 0
    while not have_posted and cur_stuff < len(stuff):
        item = stuff[cur_stuff]
        cur_stuff = cur_stuff + 1
        top = item._json
        tweet_id = top["id"]
        rawtext = top["text"]
        text = re.sub(' http.*$', '', rawtext)
        if not "entities" in top or not "media" in top["entities"]:
            continue
        media = top["entities"]["media"][0]
        media_url = media["media_url"]
        link_url = u"https://twitter.com/{}/status/{}".format(args.account, tweet_id)

        path = urlparse.urlparse(media_url).path
        ext = os.path.splitext(path)[1]
        local_media = "temp_files/media_file{}".format(ext)
        final1_media = "temp_files/final1_file{}".format(ext)
        final2_media = "temp_files/final2_file{}".format(ext)

        urllib.urlretrieve(media_url, local_media)

        # first get model ready
        if model is None and args.model is not None:
            print('Loading saved model...')
            model = Model(load(args.model).algorithm.cost)

        # get attributes
        if smile_offset is None and args.anchor_offset is not None:
            offsets = get_json_vectors(args.anchor_offset)
            dim = len(offsets[0])
            smile_offset = offset_from_string("31", offsets, dim)

        result = do_convert(local_media, final1_media, final2_media, model, smile_offset)

        media_id1 = api1.media_upload(final1_media).media_id_string
        media_id2 = api2.media_upload(final2_media).media_id_string
        # update_text = u".@{} {}".format(args.account, text)
        # update_text = text
        update_text = ""
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
                status = api2.update_status(status=update_text, media_ids=[media_id2])
                posted_id = status.id
                posted_name = status.user.screen_name
                print(u"--> Posted: {} ({} -> {})".format(update_text, posted_name, posted_id))

                status = api1.update_status(status=update_text, media_ids=[media_id1])
                posted_id = status.id
                posted_name = status.user.screen_name
                print(u"--> Posted: {} ({} -> {})".format(update_text, posted_name, posted_id))

                have_posted = True
            else:
                print(u"--> Skipped: {}".format(update_text))

