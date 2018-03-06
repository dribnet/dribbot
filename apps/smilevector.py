#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tweepy
from TwitterAPI import TwitterAPI
from TwitterAPI.TwitterError import TwitterConnectionError
import json
import argparse
import sys
import re
import urllib
import urlparse, os
from shutil import copyfile
import os
import random
from subprocess import call
from plat.sampling import real_glob

from plat.utils import anchors_from_image, offset_from_string, vectors_from_json_filelist
from plat.grid_layout import create_mine_grid

# discgen related imports
from discgen.bin.run_classifier import create_running_graphs
from discgen.interface import DiscGenModel
from faceswap import doalign
import faceswap.core
import numpy as np
from PIL import Image
from scipy.misc import imread, imsave, imresize
import theano
import hashlib
import time

tweet_suffix = u""
# tweet_suffix = u" #test_hashtag"
# tweet_suffix = u" #nuclai16"
# tweet_suffix = u" #NeuralPuppet"

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

max_allowable_extent = 300
min_allowable_extent = 60
optimal_extent = 128
# reized input file
# resized_input_file = "temp_files/resized_input_file.png"
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
# optimal input file
optimal_input = "temp_files/optimal_input_file.png"
# optimal output file
optimal_output = "temp_files/optimal_output_file.png"
# enhanced output file
enhanced_output = "temp_files/optimal_output_file_ne1x.png"

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
# archive_resized = "input_resized.png"
archive_aligned = "aligned.png"
archive_recon = "reconstruction.png"
archive_transformed = "transformed.png"
archive_swapped = "swapped.png"
archive_final_image = "final_image.png"
archive_final_movie = "final_movie.mp4"
archive_optimal_input = "optimal_input.png"
archive_optimal_output = "optimal_output.png"
archive_enhanced_output = "enhanced_output.png"

def archive_post(subdir, posted_id, original_text, post_text, respond_text, downloaded_basename, downloaded_input, final_movie, archive_dir="archives"):
    # setup paths
    archive_dir = "{}/{}".format(archive_dir, subdir)
    archive_input_path = "{}/{}".format(archive_dir, downloaded_basename)
    archive_text_path = "{}/{}".format(archive_dir, archive_text)
    # archive_resized_path = "{}/{}".format(archive_dir, archive_resized)
    archive_aligned_path = "{}/{}".format(archive_dir, archive_aligned)
    archive_recon_path = "{}/{}".format(archive_dir, archive_recon)
    archive_transformed_path = "{}/{}".format(archive_dir, archive_transformed)
    archive_swapped_path = "{}/{}".format(archive_dir, archive_swapped)
    archive_final_image_path = "{}/{}".format(archive_dir, archive_final_image)
    archive_final_movie_path = "{}/{}".format(archive_dir, archive_final_movie)
    archive_optimal_input_path = "{}/{}".format(archive_dir, archive_optimal_input)
    archive_optimal_output_path = "{}/{}".format(archive_dir, archive_optimal_output)
    archive_enhanced_output_path = "{}/{}".format(archive_dir, archive_enhanced_output)

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
    # copyfile(resized_input_file, archive_resized_path)
    copyfile(aligned_file, archive_aligned_path)
    copyfile(recon_file, archive_recon_path)
    copyfile(transformed_file, archive_transformed_path)
    copyfile(swapped_file, archive_swapped_path)
    copyfile(final_image, archive_final_image_path)
    copyfile(final_movie, archive_final_movie_path)
    copyfile(optimal_input, archive_optimal_input_path)
    copyfile(optimal_output, archive_optimal_output_path)
    copyfile(enhanced_output, archive_enhanced_output_path)

max_initial_extent = 1024
# returns [True, movie_compatible, scale_down_ratio]
def resize_to_a_good_size(infile, outfile):
    image_array = imread(infile, mode='RGB')

    im_shape = image_array.shape
    h, w, _ = im_shape
    # assume initially movie compatible, then confirm
    movie_compatible = True

    # maximum twitter aspect ratio for a movie is 239:100
    ### DISABLING CROPPING FOR NOW
    # max_width = int(h * 230 / 100)
    # if w > max_width:
    #     offset_x = (w - max_width)/2
    #     print("cropping from {0},{1} to {2},{1}".format(w,h,max_width))
    #     image_array = image_array[:,offset_x:offset_x+max_width,:]
    #     w = max_width
    # # minimum twitter aspect ratio is maybe 1:2
    # max_height = int(w * 2)
    # if h > max_height:
    #     offset_y = (h - max_height)/2
    #     print("cropping from {0},{1} to {0},{2}".format(w,h,max_height))
    #     image_array = image_array[offset_y:offset_y+max_height,:,:]
    #     h = max_height

    scale_down = None
    if w >= h:
        if w > max_initial_extent:
            scale_down = float(max_initial_extent) / w
    else:
        if h > max_initial_extent:
            scale_down = float(max_initial_extent) / h

    if scale_down is not None:
        new_w = int(scale_down * w)
        new_h = int(scale_down * h)
    else:
        scale_down = 1.0
        new_w = w
        new_h = h

    new_w = new_w - (new_w % 4)
    new_h = new_h - (new_h % 4)

    # aspect ratio must be at least 3/2
    if new_w < 1.5 * new_h:
        movie_compatible = False
    # maximum twitter aspect ratio for a movie is 239:100
    elif new_w > 2.3 * new_h:
        movie_compatible = False

    print("resizing from {},{} to {},{}".format(w, h, new_w, new_h))
    image_array_resized = imresize(image_array, (new_h, new_w))
    imsave(outfile, image_array_resized)
    return True, movie_compatible, scale_down

def resize_to_optimal(infile, scale_ratio, rect, outfile):
    image_array = imread(infile, mode='RGB')
    im_shape = image_array.shape
    h, w, _ = im_shape

    width = float(rect.right()-rect.left())
    scale_amount = (optimal_extent * scale_ratio) / width
    new_w = int(scale_amount * w)
    new_h = int(scale_amount * h)
    new_w = new_w - (new_w % 4)
    new_h = new_h - (new_h % 4)

    print("optimal resize of width {} and ratio {} went from {},{} to {},{}".format(width, scale_ratio, w, h, new_w, new_h))
    new_shape = (new_h, new_w)
    image_array_resized = imresize(image_array, new_shape)
    imsave(outfile, image_array_resized)
    return new_shape

def check_movie_compatible(shape):
    h, w = shape
    # aspect ratio must be at least 3/2
    if w < 1.5 * h:
        return False
    # maximum twitter aspect ratio for a movie is 239:100
    elif w > 2.3 * h:
        return False
    else:
        return True

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def enhance_optimal_output():
    command = "/usr/local/anaconda2/envs/enhance/bin/python ../neural-enhance3/enhance.py temp_files/optimal_output_file.png --model dlib_256_neupup1 --zoom 1 --device gpu1"
    result = os.system(command)
    if result != 0:
        # failure
        return False
    # command = "convert temp_files/optimal_output_file_ne2x.png -resize 50% temp_files/enhanced.png"
    # result = os.system(command)
    # if result != 0:
    #     # failure
    #     return False
    return True

# either returns [False, False, False] (failure) or
# [True, anchor_index, smile_detected, movie_compatible]
def do_convert(raw_infile, outfile, dmodel, classifier, do_smile, smile_offsets, image_size, initial_steps=10, recon_steps=10, offset_steps=20, optimal_steps=10, end_bumper_steps=10, check_extent=True, wraparound=True):
    failure_return_status = False, False, False, False

    # infile = resized_input_file;

    # did_resize, movie_compatible, scale_ratio = resize_to_a_good_size(raw_infile, infile)
    # if not did_resize:
    #     return failure_return_status

    # first align input face to canonical alignment and save result
    try:
        did_align, align_rect = doalign.align_face(raw_infile, aligned_file, image_size, max_extension_amount=0, min_span=72)
        width = align_rect.right()-align_rect.left()
        print("did_align, rect, width:{},{},{}".format(did_align, align_rect, width))
        if not did_align:
            return failure_return_status
    except Exception as e:
        # get_landmarks strangely fails sometimes (see bad_shriek test image)
        print("faceswap: doalign failure {}".format(e))
        return failure_return_status

    # save optimally scaled input
    optimal_shape = resize_to_optimal(raw_infile, 1.0, align_rect, optimal_input)
    infile = optimal_input;
    movie_compatible = check_movie_compatible(optimal_shape)

    # go ahead and cache the main (body) image and landmarks, and fail if face is too big
    try:
        body_image_array = imread(infile, mode='RGB')
        print(body_image_array.shape)
        body_rect, body_landmarks = faceswap.core.get_landmarks(body_image_array)
        max_extent = faceswap.core.get_max_extent(body_landmarks)
    except faceswap.core.NoFaces:
        print("faceswap: no faces in {}".format(infile))
        return failure_return_status
    except faceswap.core.TooManyFaces:
        print("faceswap: too many faces in {}".format(infile))
        return failure_return_status
    if check_extent and max_extent > max_allowable_extent:
        print("face too large: {}".format(max_extent))
        return failure_return_status
    elif check_extent and max_extent < min_allowable_extent:
        print("face to small: {}".format(max_extent))
        return failure_return_status
    else:
        print("face not too large: {}".format(max_extent))

    # read in aligned file to image array
    _, _, anchor_images = anchors_from_image(aligned_file, image_size=(image_size, image_size))

    # encode aligned image array as vector, apply offset
    encoded = dmodel.encode_images(anchor_images)[0]

    deblur_vector = smile_offsets[0]
    # randint is inclusive and blur is [0], so subtract 2
    anchor_index = random.randint(0, len(smile_offsets) - 2)
    smile_vector = smile_offsets[anchor_index+1]
    smile_score = np.dot(smile_vector, encoded)
    smile_detected = (smile_score > 0)
    print("Attribute vector detector for {}: {} {}".format(anchor_index, smile_score, smile_detected))

    if do_smile is not None:
        apply_smile = str2bool(do_smile)
    else:
        apply_smile = not smile_detected

    if apply_smile:
        print("Adding attribute {}".format(anchor_index))
        chosen_anchor = [encoded, encoded + smile_vector + deblur_vector]
    else:
        print("Removing attribute {}".format(anchor_index))
        chosen_anchor = [encoded, encoded - smile_vector + deblur_vector]

    z_dim = dmodel.get_zdim()

    # TODO: fix variable renaming
    anchors, samples_sequence_dir, movie_file = chosen_anchor, sequence_dir, outfile

    # these are the output png files
    samples_sequence_filename = samples_sequence_dir + generic_sequence

    # prepare output directory
    make_or_cleanup(samples_sequence_dir)

    # generate latents from anchors
    z_latents = create_mine_grid(rows=1, cols=offset_steps, dim=z_dim, space=offset_steps-1, anchors=anchors, spherical=True, gaussian=False)
    samples_array = dmodel.sample_at(z_latents)
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
        # face_landmarks = faceswap.core.get_landmarks(face_image_array)
        # faceswap.core.do_faceswap_from_face(infile, face_image_array, face_landmarks, swapped_file)
        faceswap.core.do_faceswap(infile, recon_file, swapped_file)
        print("swapped file: {}".format(swapped_file))
        recon_array = imread(swapped_file, mode='RGB')
    except faceswap.core.NoFaces:
        print("faceswap: no faces when generating swapped file {}".format(infile))
        imsave(debug_file, face_image_array)
        return failure_return_status
    except faceswap.core.TooManyFaces:
        print("faceswap: too many faces in {}".format(infile))
        return failure_return_status

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
            face_rect, face_landmarks = faceswap.core.get_landmarks(face_image_array)
            filename = samples_sequence_filename.format(cur_index)
            imsave(transformed_file, face_image_array)
            # faceswap.core.do_faceswap_from_face(infile, face_image_array, face_landmarks, filename)
            faceswap.core.do_faceswap(infile, transformed_file, filename)
            print("generated file: {}".format(filename))
        except faceswap.core.NoFaces:
            print("faceswap: no faces in {}".format(infile))
            return failure_return_status
        except faceswap.core.TooManyFaces:
            print("faceswap: too many faces in {}".format(infile))
            return failure_return_status

    # save optimal swapped output
    faceswap.core.do_faceswap(infile, transformed_file, optimal_output)
    if not enhance_optimal_output():
        return failure_return_status

    last_sequence_index = initial_steps + recon_steps + offset_steps - 1
    last_filename = samples_sequence_filename.format(last_sequence_index)
    copyfile(last_filename, final_image)

    final_recon_array = imread(final_image, mode='RGB')
    optimal_recon_array = imread(enhanced_output, mode='RGB')
    # now save interpolations to optimal
    for i in range(0,optimal_steps):
        frac_orig = ((optimal_steps - i) / (1.0 * optimal_steps))
        frac_optimal = (i / (1.0 * optimal_steps))
        interpolated_im = frac_orig * final_recon_array + frac_optimal * optimal_recon_array
        filename = samples_sequence_filename.format(i+last_sequence_index+1)
        imsave(filename, interpolated_im)
        print("optimal interpolated file: {}".format(filename))

    if wraparound:
        # copy last image back around to first
        first_filename = samples_sequence_filename.format(0)
        print("wraparound file: {} -> {}".format(enhanced_output, first_filename))
        copyfile(enhanced_output, first_filename)

    last_optimal_index = initial_steps + recon_steps + offset_steps + optimal_steps - 1

    # also add a final out bumper
    for i in range(last_optimal_index, last_optimal_index + end_bumper_steps):
        filename = samples_sequence_filename.format(i + 1)
        copyfile(enhanced_output, filename)
        print("end bumper file: {}".format(filename))

    # convert everything to width 640
    # resize and add fakemarks
    resize_command = "/usr/bin/convert -resize 640x {} {}".format(enhanced_output, enhanced_output)
    copy_comp = "/usr/bin/composite -gravity SouthEast -geometry +5+5 fakemark.png {} {}".format(enhanced_output, enhanced_output)
    os.system(resize_command)
    os.system(copy_comp)
    for i in range(0, last_optimal_index + end_bumper_steps + 1):
        filename = samples_sequence_filename.format(i)
        resize_command = "/usr/bin/convert -resize 640x {} {}".format(filename, filename)
        copy_comp = "/usr/bin/composite -gravity SouthEast -geometry +5+5 fakemark.png {} {}".format(filename, filename)
        os.system(resize_command)
        os.system(copy_comp)

    if os.path.exists(movie_file):
        os.remove(movie_file)
    command = "/usr/bin/ffmpeg -r 20 -f image2 -i \"{}\" -vf \"scale='min(1024,iw)':-2\" -c:v libx264 -crf 20 -pix_fmt yuv420p -tune fastdecode -y -tune zerolatency -profile:v baseline {}".format(ffmpeg_sequence_filename, movie_file)
    print("ffmpeg command: {}".format(command))
    result = os.system(command)
    if result != 0:
        return failure_return_status
    if not os.path.isfile(movie_file):
        return failure_return_status

    return True, anchor_index, smile_detected, movie_compatible

# throws exeption if things don't go well
class TwitterAPIFail(Exception):
    pass

def check_status(r):
    if r.status_code < 200 or r.status_code > 299:
        print("---> TWIITER API FAIL <---")
        print(r.status_code)
        print(r.text)
        raise TwitterAPIFail

def check_lazy_initialize(args, dmodel, classifier, vector_offsets):
    # debug: don't load anything...
    # return dmodel, classifier, smile_offsets

    # first get model ready
    if dmodel is None and args.model is not None:
        print('Loading saved model...')
        dmodel = DiscGenModel(filename=args.model)

    # first get model ready
    # if classifier is None and args.classifier is not None:
    #     print('Loading saved classifier...')
    #     classifier = create_running_graphs(args.classifier)

    # get attributes
    if vector_offsets is None and args.anchor_offset is not None:
        offsets = vectors_from_json_filelist(real_glob(args.anchor_offset))
        dim = len(offsets[0])
        offset_indexes = args.anchor_indexes.split(",")
        vector_offsets = [ -1 * offset_from_string(offset_indexes[0], offsets, dim) ]
        for i in range(len(offset_indexes) - 1):
            vector_offsets.append(offset_from_string(offset_indexes[i+1], offsets, dim))

    return dmodel, classifier, vector_offsets

if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description='Follow account and repost munged images')
    parser.add_argument('-a','--accounts', help='Accounts to follow (comma separated)', default="peopleschoice,NPG")
    parser.add_argument('-d','--debug', help='Debug: do not post', default=False, action='store_true')
    parser.add_argument('-o','--open', help='Open image (when in debug mode)', default=False, action='store_true')
    parser.add_argument('-c','--creds', help='Twitter json credentials1 (smile)', default='creds.json')
    parser.add_argument('--do-smile', default=None,
                        help='Force smile on/off (skip classifier) [1/0]')
    parser.add_argument('-n','--no-update', dest='no_update',
            help='Do not update postion on timeline', default=False, action='store_true')
    parser.add_argument('--post-image', dest='post_image',
            help='Always post an image (not a movie)', default=False, action='store_true')
    parser.add_argument("--input-file", dest='input_file', default=None,
                        help="single image file input (for debugging)")
    parser.add_argument("--archive-subdir", dest='archive_subdir', default=None,
                        help="specific subdirectory for archiving results")
    parser.add_argument("--model", dest='model', type=str, default=None,
                        help="path to the saved model")
    parser.add_argument('--anchor-offset', dest='anchor_offset', default=None,
                        help="use json file as source of each anchors offsets")
    parser.add_argument('--anchor-indexes', dest='anchor_indexes', default="0,1,2",
                        help="blur_index,smile_index,surprise_index,...")
    parser.add_argument('--anchor-text', dest='anchor_text', 
                        default=u"😀,😠,👓",
                        help="smile_emoji,surprise_emoji,...")
    parser.add_argument("--image-size", dest='image_size', type=int, default=64,
                        help="size of (offset) images")
    parser.add_argument('--classifier', dest='classifier', type=str,
                        default=None)
    parser.add_argument('--no-wrap', dest="wraparound", default=True,
                        help='Do not wraparound last image to front', action='store_false')
    args = parser.parse_args()

    # initialize and then lazily load
    dmodel = None
    classifier = None
    smile_offsets = None

    final_movie = "temp_files/final_movie.mp4"
    final_image = "temp_files/final_image.png"

    if args.archive_subdir:
        archive_subdir = args.archive_subdir
    else:
        archive_subdir = time.strftime("%Y%m%d_%H%M%S")

    # do debug as a special case
    if args.input_file:
        dmodel, classifier, smile_offsets = check_lazy_initialize(args, dmodel, classifier, smile_offsets)
        result, anchor_index, had_smile, movie_compatible = do_convert(args.input_file, final_movie, dmodel, classifier, args.do_smile, smile_offsets, args.image_size, check_extent=False, wraparound=args.wraparound)

        print("result: {}, anchor_index: {}, had_attribute: {}, movie_compatible: {}".format(result, anchor_index, had_smile, movie_compatible))
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
        text = rawtext.strip()
        text = re.sub(' http.*$', '', text)
        text = re.sub('\n.*', '', text)
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
            dmodel, classifier, smile_offsets = check_lazy_initialize(args, dmodel, classifier, smile_offsets)

            result, anchor_index, had_smile, movie_compatible = do_convert(downloaded_input, final_movie, dmodel, classifier, args.do_smile, smile_offsets, args.image_size)
            print("result: {}, anchor_index: {}, had_attribute: {}, movie_compatible: {}".format(result, anchor_index, had_smile, movie_compatible))

            blurbs = args.anchor_text.split(",")
            if had_smile:
                post_text = u"{}⬇{}".format(blurbs[anchor_index], tweet_suffix)
            else:
                post_text = u"{}⬆{}".format(blurbs[anchor_index], tweet_suffix)

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
                        if not args.post_image and movie_compatible:
                            total_bytes = os.path.getsize(final_movie)
                            file = open(final_movie, 'rb')
                            r = twitter_api.request('media/upload', {'command':'INIT', 'media_type':'video/mp4', 'total_bytes':total_bytes})
                        else:
                            total_bytes = os.path.getsize(enhanced_output)
                            file = open(enhanced_output, 'rb')
                            r = twitter_api.request('media/upload', {'command':'INIT', 'media_type':'image/png', 'total_bytes':total_bytes})
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
                    except (TwitterAPIFail, TwitterConnectionError) as e:
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


