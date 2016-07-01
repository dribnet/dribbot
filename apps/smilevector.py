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
from scipy.misc import imread, imsave, imresize
import theano
import hashlib
import time

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

def add_to_recent(infile, comment, recentfile, limit=500):
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
# reized input file
resized_input_file = "temp_files/resized_input_file.png"
# the input image file is algined and saved
aligned_file = "temp_files/aligned_file.png"
# the reconstruction is also saved
recon_file = "temp_files/recon_file.png"
# the reconstruction is also saved
transformed_file = "temp_files/transformed.png"
# reconstruction is swapped into original
swapped_file = "temp_files/swapped_file.png"
# used to save surprising failures
debug_file = "temp_files/debug.png"
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
archive_transformed = "transformed.png"
archive_swapped = "swapped.png"
archive_final_image = "final_image.png"
archive_final_movie = "final_movie.mp4"

def archive_post(subdir, posted_id, original_text, post_text, respond_text, downloaded_basename, downloaded_input, final_movie, archive_dir="archives"):
    # setup paths
    archive_dir = "{}/{}".format(archive_dir, subdir)
    archive_input_path = "{}/{}".format(archive_dir, downloaded_basename)
    archive_text_path = "{}/{}".format(archive_dir, archive_text)
    archive_aligned_path = "{}/{}".format(archive_dir, archive_aligned)
    archive_recon_path = "{}/{}".format(archive_dir, archive_recon)
    archive_transformed_path = "{}/{}".format(archive_dir, archive_transformed)
    archive_swapped_path = "{}/{}".format(archive_dir, archive_swapped)
    archive_final_image_path = "{}/{}".format(archive_dir, archive_final_image)
    archive_final_movie_path = "{}/{}".format(archive_dir, archive_final_movie)

    # prepare output directory
    make_or_cleanup(archive_dir)

    # save metadata
    with open(archive_text_path, 'a') as f:
        f.write(u"posted_id\t{}\n".format(posted_id))
        f.write(u"original_text\t{}\n".format(original_text))
        # these might be unicode. what a PITA
        f.write(u'\t'.join([u"post_text", post_text]).encode('utf-8').strip())
        f.write(u"\n")
        f.write(u'\t'.join([u"respond_text", respond_text]).encode('utf-8').strip())
        f.write(u"\n")
        f.write(u"subdir\t{}\n".format(subdir))

    # save input, a few working files, outputs
    copyfile(downloaded_input, archive_input_path)
    copyfile(aligned_file, archive_aligned_path)
    copyfile(recon_file, archive_recon_path)
    copyfile(transformed_file, archive_transformed_path)
    copyfile(swapped_file, archive_swapped_path)
    copyfile(final_image, archive_final_image_path)
    copyfile(final_movie, archive_final_movie_path)

max_extent = 480
def resize_to_a_good_size(infile, outfile):
    image_array = imread(infile)
    im_shape = image_array.shape
    if len(im_shape) == 2:
        w, h = im_shape
        print("converting from 1 channel to 3")
        image_array = np.array([image_array, image_array, image_array])
    else:
        w, h, _ = im_shape
    scale_down = None
    if w >= h:
        if w > max_extent:
            scale_down = float(max_extent) / w
    else:
        if h > max_extent:
            scale_down = float(max_extent) / h

    if scale_down is not None:
        new_w = int(scale_down * w)
        new_h = int(scale_down * h)
    else:
        new_w = w
        new_h = h

    new_w = new_w - (new_w % 4)
    new_h = new_h - (new_h % 4)

    print("resizing from {},{} to {},{}".format(w, h, new_w, new_h))
    image_array_resized = imresize(image_array, (new_w, new_h))
    imsave(outfile, image_array_resized)

def do_convert(raw_infile, outfile, model, classifier, smile_offsets, image_size, initial_steps=10, recon_steps=10, offset_steps=20, end_bumper_steps=10, check_extent=True):
    infile = resized_input_file;

    resize_to_a_good_size(raw_infile, infile)

    # first align input face to canonical alignment and save result
    if not doalign.align_face(infile, aligned_file, image_size, max_extension_amount=0):
        return False, False

    # go ahead and cache the main (body) image and landmarks, and fail if face is too big
    try:
        body_image_array = imread(infile)
        body_landmarks = faceswap.get_landmarks(body_image_array)
        max_extent = faceswap.get_max_extent(body_landmarks)
    except faceswap.NoFaces:
        print("faceswap: no faces in {}".format(infile))
        return False, False
    except faceswap.TooManyFaces:
        print("faceswap: too many faces in {}".format(infile))
        return False, False
    if check_extent and max_extent > max_allowable_extent:
        print("face to large: {}".format(max_extent))
        return False, False
    elif check_extent and max_extent < min_allowable_extent:
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
        yhat = classifier_function(anchor_images[0].reshape(1,3,image_size,image_size))
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
        chosen_anchor = [anchor[0], anchor[0] + smile_offsets[1]]
    else:
        print("Smile not detected, providing")
        chosen_anchor = [anchor[0], anchor[0] + smile_offsets[0]]

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
        # face_image_array = (255 * np.dstack(sample)).astype(np.uint8)
        face_image_array = (255 * np.dstack(sample)).astype(np.uint8)
        imsave(recon_file, face_image_array)
        # face_landmarks = faceswap.get_landmarks(face_image_array)
        # faceswap.do_faceswap_from_face(infile, face_image_array, face_landmarks, swapped_file)
        faceswap.do_faceswap(infile, recon_file, swapped_file)
        print("swapped file: {}".format(swapped_file))
        recon_array = imread(swapped_file)
    except faceswap.NoFaces:
        print("faceswap: no faces when generating swapped file {}".format(infile))
        imsave(debug_file, face_image_array)
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

    final_face_index = len(samples_array) - 1
    for i, sample in enumerate(samples_array):
        try:
            cur_index = i + initial_steps + recon_steps
            stack = np.dstack(sample)
            face_image_array = (255 * np.dstack(sample)).astype(np.uint8)
            # if i == final_face_index:
            #     imsave(transformed_file, face_image_array)
            face_landmarks = faceswap.get_landmarks(face_image_array)
            filename = samples_sequence_filename.format(cur_index)
            imsave(transformed_file, face_image_array)
            # faceswap.do_faceswap_from_face(infile, face_image_array, face_landmarks, filename)
            faceswap.do_faceswap(infile, transformed_file, filename)
            print("generated file: {}".format(filename))
        except faceswap.NoFaces:
            print("faceswap: no faces in {}".format(infile))
            return False, False
        except faceswap.TooManyFaces:
            print("faceswap: too many faces in {}".format(infile))
            return False, False

    # copy last image back around to first
    last_sequence_index = initial_steps + recon_steps + offset_steps - 1
    last_filename = samples_sequence_filename.format(last_sequence_index)
    first_filename = samples_sequence_filename.format(0)
    print("wraparound file: {} -> {}".format(last_filename, first_filename))
    copyfile(last_filename, first_filename)
    copyfile(last_filename, final_image)

    # also add a final out bumper
    for i in range(last_sequence_index, last_sequence_index + end_bumper_steps):
        filename = samples_sequence_filename.format(i + 1)
        copyfile(last_filename, filename)
        print("end bumper file: {}".format(filename))

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

# throws exeption if things don't go well
class TwitterAPIFail(Exception):
    pass

def check_status(r):
    if r.status_code < 200 or r.status_code > 299:
        print("---> TWIITER API FAIL <---")
        print(r.status_code)
        print(r.text)
        raise TwitterAPIFail

def check_lazy_initialize(args, model, classifier, smile_offsets):
    # first get model ready
    if model is None and args.model is not None:
        print('Loading saved model...')
        model = Model(load(args.model).algorithm.cost)

    # first get model ready
    if classifier is None and args.classifier is not None:
        print('Loading saved classifier...')
        classifier = create_running_graphs(args.classifier)

    # get attributes
    if smile_offsets is None and args.anchor_offset is not None:
        offsets = get_json_vectors(args.anchor_offset)
        dim = len(offsets[0])
        smile_offset_smile = offset_from_string("31", offsets, dim)
        smile_offset_open = offset_from_string("21", offsets, dim)
        smile_offset_blur = offset_from_string("10", offsets, dim)
        pos_smile_offset = 0.75 * smile_offset_open + 0.75 * smile_offset_smile - 2.0 * smile_offset_blur
        neg_smile_offset = -1 * smile_offset_open - smile_offset_smile - 2.0 * smile_offset_blur
        smile_offsets = [pos_smile_offset, neg_smile_offset]

    return model, classifier, smile_offsets

if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description='Follow account and repost munged images')
    parser.add_argument('-a','--accounts', help='Accounts to follow (comma separated)', default="peopleschoice,NPG")
    parser.add_argument('-d','--debug', help='Debug: do not post', default=False, action='store_true')
    parser.add_argument('-o','--open', help='Open image (when in debug mode)', default=False, action='store_true')
    parser.add_argument('-c','--creds', help='Twitter json credentials1 (smile)', default='creds.json')
    parser.add_argument('-n','--no-update', dest='no_update',
            help='Do not update postion on timeline', default=False, action='store_true')
    parser.add_argument("--input-file", dest='input_file', default=None,
                        help="single image file input (for debugging)")
    parser.add_argument("--archive-subdir", dest='archive_subdir', default=None,
                        help="specific subdirectory for archiving results")
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
    smile_offsets = None

    final_movie = "temp_files/final_movie.mp4"

    if args.archive_subdir:
        archive_subdir = args.archive_subdir
    else:
        archive_subdir = time.strftime("%Y%m%d_%H%M%S")

    # do debug as a special case
    if args.input_file:
        model, classifier, smile_offsets = check_lazy_initialize(args, model, classifier, smile_offsets)
        result, had_smile = do_convert(args.input_file, final_movie, model, classifier, smile_offsets, args.image_size, check_extent=False)
        print("result: {}, had_smile: {}".format(result, had_smile))
        if result and not args.no_update:
            input_basename = os.path.basename(args.input_file)
            archive_post(archive_subdir, "no_id", had_smile, "no_post", "no_respond", input_basename, args.input_file, final_movie, "debug")
        exit(0)

    # state tracking files from run to run
    recentfile = "temp_files/recent_posts.txt"

    # now fire up tweepy
    with open(args.creds) as data_file:
        creds = json.load(data_file)

    auth = tweepy.OAuthHandler(creds["consumer_key"], creds["consumer_secret"])
    auth.set_access_token(creds["access_token"], creds["access_token_secret"])
    tweepy_api = tweepy.API(auth)

    twitter_api = TwitterAPI(creds["consumer_key"], creds["consumer_secret"], creds["access_token"], creds["access_token_secret"])

    accounts = args.accounts.split(",")
    account = random.choice(accounts)
    print("choosing a post from account {}".format(account))

    # just grab most recent tweet
    stuff = tweepy_api.user_timeline(screen_name = account, \
        count = 200, \
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
        original_text = text.encode('ascii', 'ignore')
        post_text = u"no post"

        if not "entities" in top or not "media" in top["entities"]:
            continue

        print(u"Looking at post: {}".format(original_text))

        media = top["entities"]["media"][0]
        media_url = media["media_url"]
        link_url = u"https://twitter.com/{}/status/{}".format(account, tweet_id)

        path = urlparse.urlparse(media_url).path
        ext = os.path.splitext(path)[1]
        downloaded_basename = "input_image{}".format(ext)
        downloaded_input = "temp_files/{}".format(downloaded_basename)

        print("Downloading {} as {}".format(media_url, downloaded_input))
        urllib.urlretrieve(media_url, downloaded_input)

        result = check_recent(downloaded_input, recentfile)
        if result is False:
            print "Image found in recent cache, skipping"
        else:
            model, classifier, smile_offsets = check_lazy_initialize(args, model, classifier, smile_offsets)

            result, had_smile = do_convert(downloaded_input, final_movie, model, classifier, smile_offsets, args.image_size)
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
                num_upload_failures = 0
                upload_failure_retry_delay = 30
                max_upload_failures = 10
                update_response = None

                while update_response is None:
                    try:
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

                        update_response = r
                    except TwitterAPIFail:
                        num_upload_failures = num_upload_failures + 1
                        if num_upload_failures >= max_upload_failures:
                            print("----> TWITTER FAILED {} TIMES, exiting".format(num_upload_failures))
                            exit(1)
                        print("----> TWITTER FAIL #{}: waiting {} seconds and will retry".format(num_upload_failures, upload_failure_retry_delay))
                        time.sleep(upload_failure_retry_delay)


                r_json = update_response.json()
                # note - setting posted_id exits the loop
                posted_id = r_json['id']
                posted_name = r_json['user']['screen_name']

                print(u"--> Updated: {} ({} -> {})".format(original_text, posted_name, posted_id))
                respond_text = u"@{} reposted from: {}".format(posted_name, link_url)
                status = tweepy_api.update_status(status=respond_text, in_reply_to_status_id=posted_id)
            else:
                print(u"--> Skipped: {}".format(original_text))

        if posted_id is not None and not args.no_update:
            print("updating state and archiving")
            add_to_recent(downloaded_input, original_text, recentfile)
            if posted_id is not None:
                archive_post(archive_subdir, posted_id, original_text, post_text, respond_text, downloaded_basename, downloaded_input, final_movie)
        else:
            print("(update skipped)")


