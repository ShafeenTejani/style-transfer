# Copyright (c) 2016-2017 Shafeen Tejani. Released under GPLv3.

import os

import numpy as np
import scipy.misc
from os.path import exists
from sys import stdout

from style_transfer import StyleTransfer

import math
from argparse import ArgumentParser

# default arguments
CONTENT_WEIGHT = 5e1
STYLE_WEIGHT = 1e2
TV_WEIGHT = 1e2
LEARNING_RATE = 1e1
ITERATIONS = 1000
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content',
            dest='content', help='content image',
            metavar='CONTENT', required=True)
    parser.add_argument('--style',
            dest='style', help='one or more style images',
            metavar='STYLE', required=True)
    parser.add_argument('--output',
            dest='output', help='output path',
            metavar='OUTPUT', required=True)
    parser.add_argument('--iterations', type=int,
            dest='iterations', help='iterations (default %(default)s)',
            metavar='ITERATIONS', default=ITERATIONS)
    parser.add_argument('--print-iterations', type=int,
            dest='print_iterations', help='statistics printing frequency',
            metavar='PRINT_ITERATIONS')
    parser.add_argument('--checkpoint-output',
            dest='checkpoint_output', help='checkpoint output format, e.g. output%%s.jpg',
            metavar='OUTPUT')
    parser.add_argument('--checkpoint-iterations', type=int,
            dest='checkpoint_iterations', help='checkpoint frequency',
            metavar='CHECKPOINT_ITERATIONS')
    parser.add_argument('--content-weight', type=float,
            dest='content_weight', help='content weight (default %(default)s)',
            metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)
    parser.add_argument('--style-weight', type=float,
            dest='style_weight', help='style weight (default %(default)s)',
            metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)
    parser.add_argument('--tv-weight', type=float,
            dest='tv_weight', help='total variation regularization weight (default %(default)s)',
            metavar='TV_WEIGHT', default=TV_WEIGHT)
    parser.add_argument('--learning-rate', type=float,
            dest='learning_rate', help='learning rate (default %(default)s)',
            metavar='LEARNING_RATE', default=LEARNING_RATE)
    parser.add_argument('--initial',
            dest='initial', help='initial image',
            metavar='INITIAL')
    parser.add_argument('--use-gpu', dest='use_gpu', help='run on a GPU', action='store_true')
    parser.set_defaults(use_gpu=False)
    return parser


def load_image(image_path):
    assert exists(image_path), "image {} does not exist".format(image_path)
    img = imread(image_path)
    img = img.astype("float32")
    img = np.ndarray.reshape(img, (1,) + img.shape)
    return img

def main():
    parser = build_parser()
    options = parser.parse_args()

    if not os.path.isfile(VGG_PATH):
        parser.error("Network %s does not exist." % VGG_PATH)

    content_image = load_image(options.content)
    style_image = load_image(options.style)

    initial = options.initial
    if initial is not None:
        initial = scipy.misc.imresize(imread(initial), content_image.shape[:2])

    if options.checkpoint_output and "%s" not in options.checkpoint_output:
        parser.error("To save intermediate images, the checkpoint output "
                     "parameter must contain `%s` (e.g. `foo%s.jpg`)")

    device = '/gpu:0' if options.use_gpu else '/cpu:0'

    style_transfer = StyleTransfer(
        vgg_path=VGG_PATH,
        content=content_image,
        style=style_image,
        content_weight=options.content_weight,
        style_weight=options.style_weight,
        tv_weight=options.style_weight,
        initial=initial,
        device=device)

    for iteration, image, losses in style_transfer.train(
        learning_rate=options.learning_rate,
        iterations=options.iterations,
        checkpoint_iterations=options.checkpoint_iterations
    ):
        print_losses(losses)

        output_file = None
        if iteration is not None:
            if options.checkpoint_output:
                output_file = options.checkpoint_output % iteration
        else:
            output_file = options.output
        if output_file:
            imsave(output_file, image)


def print_losses(losses):
    stdout.write('  content loss: %g\n' % losses['content'])
    stdout.write('    style loss: %g\n' % losses['style'])
    stdout.write('       tv loss: %g\n' % losses['total_variation'])
    stdout.write('    total loss: %g\n' % losses['total'])

def imread(path):
    return scipy.misc.imread(path).astype(np.float)


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)


if __name__ == '__main__':
    main()
