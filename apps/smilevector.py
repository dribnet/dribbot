#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tweepy
from TwitterAPI import TwitterAPI
import json
import argparse
import sys
import re
import urllib
import urlparse, os
from shutil import copyfile
import os
import doalign
import random
from subprocess import call

# discgen related imports
from blocks.model import Model
from blocks.serialization import load
from blocks.select import Selector
from utils.sample_utils import offset_from_string, anchors_from_image, get_image_vectors, compute_splash, get_json_vectors
from utils.sample import samples_from_latents
from experiments.run_classifier import create_running_graphs
import faceswap
import numpy as np
from PIL import Image
from scipy.misc import imread, imsave
import theano

def check_recent(infile, recentfile):
    return True

def add_to_recent(infile, recentfile, limit=48):
    return True

max_allowable_extent = 140

def do_convert(infile, outfile1, outfile2, model, classifier, smile_offset, image_size, initial_steps=10, recon_steps=10, offset_steps=20):
    aligned_file = "temp_files/aligned_file.png"
    recon_file = "temp_files/recon_file.png"
    smile1_dir = "temp_files/smile1_seq/"
    smile2_dir = "temp_files/smile2_seq/"
    generic_sequence = "{:03d}.png"
    ffmpeg_sequence = "%3d.png"

    # first try to align the face
    if not doalign.align_face(local_media, aligned_file, image_size):
        return False, False

    # go ahead and cache the main (body) image and landmarks, and fail if face is too big
    body_image_array = imread(infile)
    body_landmarks = faceswap.get_landmarks(body_image_array)
    max_extent = faceswap.get_max_extent(body_landmarks)
    if (max_extent > max_allowable_extent):
        print("face to large: {}", max_extent)
        return False, False
    else:
        print("face not too large: {}", max_extent)

    # first encode image to vector
    _, _, anchor_images = anchors_from_image(aligned_file, image_size=(image_size, image_size))

    # classifiy
    classifier_function = None
    if classifier != None:
        print('Compiling classifier function...')
        classifier_function = theano.function(classifier.inputs, classifier.outputs)
        yhat = classifier_function(anchor_images[0].reshape(1,3,256,256))
        yn = np.array(yhat[0])
        has_smile = False
        if(yn[0][31] >= 0.5):
            has_smile = True
        print("RESULT 31=smile", yn[0][31], has_smile)
    else:
        has_smile = random.choice([True, False])

    anchor = get_image_vectors(model, anchor_images)
    if has_smile:
        print("Smile detected, removing")
        chosen_anchor = [anchor[0], anchor[0] - smile_offset]
    else:
        print("Smile not detected, providing")
        chosen_anchor = [anchor[0], anchor[0] + smile_offset]

    # fire up decoder
    selector = Selector(model.top_bricks)
    decoder_mlp, = selector.select('/decoder_mlp').bricks
    z_dim = decoder_mlp.input_dim

    settings = [
        [chosen_anchor, smile1_dir, outfile1]
    ]

    for cur_setting in settings:
        anchors, samples_sequence_dir, movie_file = cur_setting

        # these are the output png files
        samples_sequence_filename = samples_sequence_dir + generic_sequence

        # make output directory if it is not there
        if not os.path.exists(samples_sequence_dir):
            os.makedirs(samples_sequence_dir)

        # and clean it out if it is there
        filelist = [ f for f in os.listdir(samples_sequence_dir) if f.endswith(".png") ]
        for f in filelist:
            os.remove(os.path.join(samples_sequence_dir, f))

        # generate latents from anchors
        z_latents = compute_splash(rows=1, cols=offset_steps, dim=z_dim, space=offset_steps-1, anchors=anchors, spherical=True, gaussian=False)
        samples_array = samples_from_latents(z_latents, model)

        # save original file as-is
        for i in range(initial_steps):
            filename = samples_sequence_filename.format(1 + i)
            imsave(filename, body_image_array)
            print("original file: {}".format(filename))

        # build face swapped reconstruction
        sample = samples_array[0]
        try:
            face_image_array = (255 * np.dstack(sample)).astype(np.uint8)
            face_landmarks = faceswap.get_landmarks(face_image_array)
            faceswap.do_faceswap_from_face(infile, face_image_array, face_landmarks, recon_file)
            print("recon file: {}".format(recon_file))
            recon_array = imread(recon_file)
        except faceswap.NoFaces:
            print("faceswap: no faces in {}".format(infile))
            return False, False
        except faceswap.TooManyFaces:
            print("faceswap: too many faces in {}".format(infile))
            return False, False

        # now save interpolations to recon
        for i in range(1,recon_steps):
            frac_orig = ((recon_steps - i) / (1.0 * recon_steps))
            frac_recon = (i / (1.0 * recon_steps))
            interpolated_im = frac_orig * body_image_array + frac_recon * recon_array
            filename = samples_sequence_filename.format(i+initial_steps)
            imsave(filename, interpolated_im)
            print("interpolated file: {}".format(filename))

        for i, sample in enumerate(samples_array):
            try:
                cur_index = i + initial_steps + recon_steps
                stack = np.dstack(sample)
                face_image_array = (255 * np.dstack(sample)).astype(np.uint8)
                face_landmarks = faceswap.get_landmarks(face_image_array)
                filename = samples_sequence_filename.format(cur_index)
                faceswap.do_faceswap_from_face(infile, face_image_array, face_landmarks, filename)
                print("generated file: {}".format(filename))
            except faceswap.NoFaces:
                print("faceswap: no faces in {}".format(infile))
                return False, False
            except faceswap.TooManyFaces:
                print("faceswap: too many faces in {}".format(infile))
                return False, False

        # copy last image back around to first
        last_filename = samples_sequence_filename.format(initial_steps + recon_steps + offset_steps - 1)
        first_filename = samples_sequence_filename.format(0)
        print("wraparound file: {} -> {}".format(last_filename, first_filename))
        copyfile(last_filename, first_filename)

        cur_ffmpeg_sequence = samples_sequence_dir + ffmpeg_sequence
        if os.path.exists(movie_file):
            os.remove(movie_file)
        command = "/usr/local/bin/ffmpeg -r 20 -f image2 -i \"{}\" -c:v libx264 -crf 20 -pix_fmt yuv420p -tune fastdecode -y -tune zerolatency -profile:v baseline {}".format(cur_ffmpeg_sequence, movie_file)
        print("COMMAND IS ", command)
        result = os.system(command)
        if result != 0:
            return False, False
        if not os.path.isfile(movie_file):
            return False, False

    return True, has_smile

def check_status(r):
    if r.status_code < 200 or r.status_code > 299:
        print(r.status_code)
        print(r.text)
        sys.exit(1)
#
#

if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description='Follow account and repost munged images')
    parser.add_argument('-a','--account', help='Account to follow', default="peopleschoice")
    parser.add_argument('-d','--debug', help='Debug: do not post', default=False, action='store_true')
    parser.add_argument('-o','--open', help='Open image (when in debug mode)', default=False, action='store_true')
    parser.add_argument('-s','--single', help='Process only a single image', default=False, action='store_true')
    parser.add_argument('-c','--creds', help='Twitter json credentials1 (smile)', default='forcedsmilebot.json')
    parser.add_argument('-n','--no-update', dest='no_update',
            help='Do not update postion on timeline', default=False, action='store_true')
    parser.add_argument("--model", dest='model', type=str, default=None,
                        help="path to the saved model")
    parser.add_argument('--anchor-offset', dest='anchor_offset', default=None,
                        help="use json file as source of each anchors offsets")
    parser.add_argument("--image-size", dest='image_size', type=int, default=64,
                        help="size of (offset) images")
    parser.add_argument('--classifier', dest='classifier', type=str,
                        default=None)
    args = parser.parse_args()

    # initialize and then lazily load
    model = None
    classifier = None
    smile_offset = None

    # state tracking files from run to run
    tempfile = "temp_files/{}_follow_account_lastid.txt".format(args.account)
    recentfile = "temp_files/{}_recent_posts_hashrecentes.txt".format(args.account)

    # try to recover last known tweet_id (if fails, last_id will stay None)
    last_id = None
    try:
        f = open(tempfile).read()
        last_id = int(f)
    except IOError:
        pass

    # now fire up tweepy
    with open(args.creds) as data_file:
        creds = json.load(data_file)

    auth1 = tweepy.OAuthHandler(creds["consumer_key"], creds["consumer_secret"])
    auth1.set_access_token(creds["access_token"], creds["access_token_secret"])
    api1 = tweepy.API(auth1)

    api_raw = TwitterAPI(creds["consumer_key"], creds["consumer_secret"], creds["access_token"], creds["access_token_secret"])

    # ready to scrape the last 100 tweets
    if last_id is None:
        # just grab most recent tweet
        stuff = api1.user_timeline(screen_name = args.account, \
            count = 100, \
            include_rts = False,
            exclude_replies = False)
    else:
        # look back up to 100 tweets since last one and then show next one
        stuff = api1.user_timeline(screen_name = args.account, \
            count = 100, \
            since_id = last_id,
            include_rts = False,
            exclude_replies = False)

    # make sure there is a result or quit
    if len(stuff) == 0:
        print("(nothing to do)")
        sys.exit(0)

    # honor command line request to only process one file
    if args.single:
        stuff = [ stuff[-1] ]

    # will update this if we actually post so we can quit
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
        final1_media = "temp_files/final1_file.mp4"
        final2_media = "temp_files/final2_file.mp4"

        urllib.urlretrieve(media_url, local_media)

        # first get model ready
        if model is None and args.model is not None:
            print('Loading saved model...')
            model = Model(load(args.model).algorithm.cost)

        # first get model ready
        if classifier is None and args.classifier is not None:
            print('Loading saved classifier...')
            classifier = create_running_graphs(args.classifier)

        # get attributes
        if smile_offset is None and args.anchor_offset is not None:
            offsets = get_json_vectors(args.anchor_offset)
            dim = len(offsets[0])
            smile_offset = offset_from_string("31", offsets, dim)

        result = check_recent(local_media, recentfile)
        if result:
            result, had_smile = do_convert(local_media, final1_media, final2_media, model, classifier, smile_offset, args.image_size)

        if had_smile:
            post_text = u"😀⬇"
        else:
            post_text = u"😀⬆"

        # update_text = u".@{} {}".format(args.account, text)
        update_text = u"{}".format(text)
        if args.debug:
            print(u"Update text: {}, Image1: {}, Image2: {}".format(update_text, final1_media, final2_media))
            if not result:
                print("(Image conversation failed)")
                if args.open:
                    call(["open", local_media])
            else:
                if args.open:
                    call(["open", final_media])
        else:
            if result:
                # https://github.com/geduldig/TwitterAPI/blob/master/examples/upload_video.py
                bytes_sent = 0
                total_bytes = os.path.getsize(final1_media)
                file = open(final1_media, 'rb')
                r = api_raw.request('media/upload', {'command':'INIT', 'media_type':'video/mp4', 'total_bytes':total_bytes})
                check_status(r)

                media_id = r.json()['media_id']
                segment_id = 0

                while bytes_sent < total_bytes:
                  chunk = file.read(4*1024*1024)
                  r = api_raw.request('media/upload', {'command':'APPEND', 'media_id':media_id, 'segment_index':segment_id}, {'media':chunk})
                  check_status(r)
                  segment_id = segment_id + 1
                  bytes_sent = file.tell()
                  print('[' + str(total_bytes) + ']', str(bytes_sent))

                print("FINALIZING")
                r = api_raw.request('media/upload', {'command':'FINALIZE', 'media_id':media_id})
                check_status(r)

                print("posting")
                r = api_raw.request('statuses/update', {'status':post_text, 'media_ids':media_id})
                check_status(r)

                r_json = r.json()
                # print("JSON: ", r_json)
                posted_id = r_json['id']
                posted_name = r_json['user']['screen_name']

                try:
                    print(u"--> Posted: {} ({} -> {})".format(update_text, posted_name, posted_id))
                except:
                    print("--> Something posted")
                respond_text = u"@{} reposted from: {}".format(posted_name, link_url)
                status = api1.update_status(status=respond_text, in_reply_to_status_id=posted_id)

                # media_id1 = api1.media_upload(final1_media).media_id_string
                # media_id2 = api2.media_upload(final2_media).media_id_string

                # status = api2.update_status(status=empty_text, media_ids=[media_id2])
                # posted_id = status.id
                # posted_name = status.user.screen_name
                # print(u"--> Posted: {} ({} -> {})".format(update_text, posted_name, posted_id))
                # respond_text = u"@{} reposted from: {}".format(posted_name, link_url)
                # status = api2.update_status(status=respond_text, in_reply_to_status_id=posted_id)

                # status = api1.update_status(status=empty_text, media_ids=[media_id1])
                # posted_id = status.id
                # posted_name = status.user.screen_name
                # print(u"--> Posted: {} ({} -> {})".format(update_text, posted_name, posted_id))
                # respond_text = u"@{} reposted from: {}".format(posted_name, link_url)
                # status = api1.update_status(status=respond_text, in_reply_to_status_id=posted_id)

                have_posted = True
            else:
                try:
                    print(u"--> Skipped: {}".format(update_text))
                except:
                    print("--> Something skipped")

        if have_posted and not args.no_update:
            add_to_recent(local_media, recentfile)

