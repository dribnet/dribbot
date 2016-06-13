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
import hashlib

# returns True if file not found and can be processed
def check_recent(infile, recentfile):
    try:
        with open(recentfile) as f :
            content = f.readlines()
    except EnvironmentError: # parent of IOError, OSError
        # that's ok
        print("No cache of recent files not found ({}), will create".format(recentfile))
        return True

    md5hash = hashlib.md5(open(infile, 'rb').read()).hexdigest().encode('utf-8')
    known_hashes = [line.split('\t', 1)[0] for line in content]
    if md5hash in known_hashes:
        return False
    else:
        return True

def add_to_recent(infile, comment, recentfile, limit=100):
    if os.path.isfile(recentfile):
        copyfile(recentfile, "{}.bak".format(recentfile))

    try:
        with open(recentfile) as f :
            content = f.readlines()
    except EnvironmentError: # parent of IOError, OSError
        content = []

    md5hash = hashlib.md5(open(infile, 'rb').read()).hexdigest()
    newitem = u"{}\t{}\n".format(md5hash, comment)
    content.insert(0, newitem)
    content = content[:limit]

    with open(recentfile, "w") as f:
        f.writelines(content)

max_allowable_extent = 180
min_allowable_extent = 60
# the input image file is algined and saved
aligned_file = "temp_files/aligned_file.png"
# the reconstruction is also saved
recon_file = "temp_files/recon_file.png"
# this is the final swapped image
final_image = "temp_files/final_image.png"
# the interpolated sequence is saved into this directory
sequence_dir = "temp_files/image_sequence/"
# template for output png files
generic_sequence = "{:03d}.png"
samples_sequence_filename = sequence_dir + generic_sequence
# template for ffmpeg arguments
ffmpeg_sequence = "%3d.png"
ffmpeg_sequence_filename = sequence_dir + ffmpeg_sequence

def make_or_cleanup(local_dir):
    # make output directory if it is not there
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    # and clean it out if it is there
    filelist = [ f for f in os.listdir(local_dir) ]
    for f in filelist:
        os.remove(os.path.join(local_dir, f))

archive_text = "metadata.txt"
archive_aligned = "aligned.png"
archive_recon = "reconstruction.png"
archive_final_image = "final_image.png"
archive_final_movie = "final_movie.mp4"

def archive_post(posted_id, original_text, post_text, respond_text, downloaded_basename, downloaded_input, final_movie):
    # setup paths
    archive_dir = "archives/{}".format(posted_id)
    archive_input_path = "{}/{}".format(archive_dir, downloaded_basename)
    archive_text_path = "{}/{}".format(archive_dir, archive_text)
    archive_aligned_path = "{}/{}".format(archive_dir, archive_aligned)
    archive_recon_path = "{}/{}".format(archive_dir, archive_recon)
    archive_final_iamge_path = "{}/{}".format(archive_dir, archive_final_image)
    archive_final_movie_path = "{}/{}".format(archive_dir, archive_final_movie)

    # prepare output directory
    make_or_cleanup(archive_dir)

    # save metadata
    with open(archive_text_path, 'a') as the_file:
        the_file.write(u' '.join([u"posted_id", str(posted_id)]).encode('utf-8').strip())
        the_file.write(u' '.join([u"original_text", original_text]).encode('utf-8').strip())
        the_file.write(u' '.join([u"post_text", post_text]).encode('utf-8').strip())
        the_file.write(u' '.join([u"respond_text", respond_text]).encode('utf-8').strip())

    # save input, a few working files, outputs
    copyfile(downloaded_input, archive_input_path)
    copyfile(aligned_file, archive_aligned)
    copyfile(recon_file, archive_recon_path)
    copyfile(final_image, archive_final_iamge_path)
    copyfile(final_movie, archive_final_movie_path)

def do_convert(infile, outfile, model, classifier, smile_offset, image_size, initial_steps=10, recon_steps=10, offset_steps=20):

    # first align input face to canonical alignment and save result
    if not doalign.align_face(infile, aligned_file, image_size):
        return False, False

    # go ahead and cache the main (body) image and landmarks, and fail if face is too big
    body_image_array = imread(infile)
    body_landmarks = faceswap.get_landmarks(body_image_array)
    max_extent = faceswap.get_max_extent(body_landmarks)
    if (max_extent > max_allowable_extent):
        print("face to large: {}".format(max_extent))
        return False, False
    elif (max_extent < min_allowable_extent):
        print("face to small: {}".format(max_extent))
        return False, False
    else:
        print("face not too large: {}".format(max_extent))

    # read in aligned file to image array
    _, _, anchor_images = anchors_from_image(aligned_file, image_size=(image_size, image_size))

    # classifiy aligned as smiling or not
    classifier_function = None
    if classifier != None:
        print('Compiling classifier function...')
        classifier_function = theano.function(classifier.inputs, classifier.outputs)
        yhat = classifier_function(anchor_images[0].reshape(1,3,256,256))
        yn = np.array(yhat[0])
        has_smile = False
        if(yn[0][31] >= 0.5):
            has_smile = True
        print("Smile detector:", yn[0][31], has_smile)
    else:
        has_smile = random.choice([True, False])

    # encode aligned image array as vector, apply offset
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

    # TODO: fix variable renaming
    anchors, samples_sequence_dir, movie_file = chosen_anchor, sequence_dir, outfile

    # these are the output png files
    samples_sequence_filename = samples_sequence_dir + generic_sequence

    # prepare output directory
    make_or_cleanup(samples_sequence_dir)

    # generate latents from anchors
    z_latents = compute_splash(rows=1, cols=offset_steps, dim=z_dim, space=offset_steps-1, anchors=anchors, spherical=True, gaussian=False)
    samples_array = samples_from_latents(z_latents, model)
    print("Samples array: ", samples_array.shape)

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
    copyfile(last_filename, final_image)

    if os.path.exists(movie_file):
        os.remove(movie_file)
    command = "/usr/local/bin/ffmpeg -r 20 -f image2 -i \"{}\" -c:v libx264 -crf 20 -pix_fmt yuv420p -tune fastdecode -y -tune zerolatency -profile:v baseline {}".format(ffmpeg_sequence_filename, movie_file)
    print("ffmpeg command: {}".format(command))
    result = os.system(command)
    if result != 0:
        return False, False
    if not os.path.isfile(movie_file):
        return False, False

    return True, has_smile

# TODO: this could be smarter
def check_status(r):
    if r.status_code < 200 or r.status_code > 299:
        print("---> TWIITER API FAIL <---")
        print(r.status_code)
        print(r.text)
        sys.exit(1)

if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description='Follow account and repost munged images')
    parser.add_argument('-a','--account', help='Account to follow', default="peopleschoice")
    parser.add_argument('-d','--debug', help='Debug: do not post', default=False, action='store_true')
    parser.add_argument('-o','--open', help='Open image (when in debug mode)', default=False, action='store_true')
    parser.add_argument('-c','--creds', help='Twitter json credentials1 (smile)', default='creds.json')
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
    recentfile = "temp_files/recent_posts.txt"

    # now fire up tweepy
    with open(args.creds) as data_file:
        creds = json.load(data_file)

    auth = tweepy.OAuthHandler(creds["consumer_key"], creds["consumer_secret"])
    auth.set_access_token(creds["access_token"], creds["access_token_secret"])
    tweepy_api = tweepy.API(auth)

    twitter_api = TwitterAPI(creds["consumer_key"], creds["consumer_secret"], creds["access_token"], creds["access_token_secret"])

    # just grab most recent tweet
    stuff = tweepy_api.user_timeline(screen_name = args.account, \
        count = 100, \
        include_rts = False,
        exclude_replies = False)

    # make sure there is a result or quit
    if len(stuff) == 0:
        print("(nothing to do)")
        sys.exit(0)

    # will update this if we actually post so we can quit
    posted_id = None

    # for item in reversed(stuff):
    cur_stuff = 0
    while posted_id is None and cur_stuff < len(stuff):
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
        downloaded_basename = "input_image{}".format(ext)
        downloaded_input = "temp_files/{}".format(downloaded_basename)
        final_movie = "temp_files/final_movie.mp4"

        print("Downloading {} as {}".format(media_url, downloaded_input))
        urllib.urlretrieve(media_url, downloaded_input)

        original_text = text.encode('ascii', 'ignore')
        post_text = u"no post"
        result = check_recent(downloaded_input, recentfile)
        if result is False:
            print "Image found in recent cache, skipping"
        else:
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

            result, had_smile = do_convert(downloaded_input, final_movie, model, classifier, smile_offset, args.image_size)
            if had_smile:
                post_text = u"😀⬇"
            else:
                post_text = u"😀⬆"

        if args.debug:
            print(u"Update text: {}, Movie: {}".format(original_text, final_movie))
            if not result:
                print("Not processed")
                if args.open:
                    call(["open", downloaded_input])
            else:
                posted_id = "pid_{}".format(os.getpid())
                respond_text = u"Reposted from: {}".format(link_url)
                if args.open:
                    call(["open", final_movie])
        else:
            if result:
                # https://github.com/geduldig/TwitterAPI/blob/master/examples/upload_video.py
                bytes_sent = 0
                total_bytes = os.path.getsize(final_movie)
                file = open(final_movie, 'rb')
                r = twitter_api.request('media/upload', {'command':'INIT', 'media_type':'video/mp4', 'total_bytes':total_bytes})
                check_status(r)

                media_id = r.json()['media_id']
                segment_id = 0

                while bytes_sent < total_bytes:
                  chunk = file.read(4*1024*1024)
                  r = twitter_api.request('media/upload', {'command':'APPEND', 'media_id':media_id, 'segment_index':segment_id}, {'media':chunk})
                  check_status(r)
                  segment_id = segment_id + 1
                  bytes_sent = file.tell()
                  print('[' + str(total_bytes) + ']', str(bytes_sent))

                print("finalizing movie upload")
                r = twitter_api.request('media/upload', {'command':'FINALIZE', 'media_id':media_id})
                check_status(r)

                print("sending status update")
                r = twitter_api.request('statuses/update', {'status':post_text, 'media_ids':media_id})
                check_status(r)

                r_json = r.json()
                # note - setting posted_id exits the loop
                posted_id = r_json['id']
                posted_name = r_json['user']['screen_name']

                try:
                    print(u"--> Updated: {} ({} -> {})".format(original_text, posted_name, posted_id))
                except:
                    print("--> Something updated")
                respond_text = u"@{} reposted from: {}".format(posted_name, link_url)
                status = tweepy_api.update_status(status=respond_text, in_reply_to_status_id=posted_id)
            else:
                try:
                    print(u"--> Skipped: {}".format(original_text))
                except:
                    print("--> Something skipped")

        if posted_id is not None and not args.no_update:
            print("updating state and archiving")
            add_to_recent(downloaded_input, original_text, recentfile)
            if posted_id is not None:
                archive_post(posted_id, original_text, post_text, respond_text, downloaded_basename, downloaded_input, final_movie)
        else:
            print("(update skipped)")


