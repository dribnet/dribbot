import tweepy
import json
import argparse
import sys
import re
import urllib
import urlparse, os
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

def do_convert(infile, outfile, model, smile_offset):
    aligned_file = "temp_files/aligned_file.png"
    smile_file = "temp_files/smile_file.png"

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
    z = z + smile_offset
    grid_from_latents(z, model, rows=1, cols=1, anchor_images=anchor_images, tight=True, shoulders=False, save_path=smile_file)

    try:
        faceswap.do_faceswap(infile, smile_file, outfile)
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
    parser.add_argument('-n','--no-update', dest='no_update',
            help='Do not update postion on timeline', default=False, action='store_true')
    parser.add_argument("--model", dest='model', type=str, default=None,
                        help="path to the saved model")
    parser.add_argument('--anchor-offset', dest='anchor_offset', default=None,
                        help="use json file as source of each anchors offsets")

    args = parser.parse_args()

    # now fire up tweepy
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
            include_rts = False,
            exclude_replies = False)
    else:
        # look back up to 100 tweets since last one and then show next one
        stuff = api.user_timeline(screen_name = args.account, \
            count = 100, \
            since_id = last_id,
            include_rts = False,
            exclude_replies = False)

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

    # first get model ready
    if args.model is not None:
        print('Loading saved model...')
        model = Model(load(args.model).algorithm.cost)

    # get attributes
    if args.anchor_offset is not None:
        offsets = get_json_vectors(args.anchor_offset)
        dim = len(offsets[0])
        smile_offset = offset_from_string("31", offsets, dim)

    result = do_convert(local_media, final_media, model, smile_offset)

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
