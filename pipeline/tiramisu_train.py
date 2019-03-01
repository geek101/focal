"""
Bablu
Train using Tiramisu

Copyright (c) 2019 Powell Molleti.
Licensed under the MIT License (see LICENSE for details)
"""

import sys
import argparse
import logging
import os

from deps.Tiramisu.model import DenseTiramisu


def get_argparser():
    parser = argparse.ArgumentParser(description='Tiramisu trainer')
    parser.add_argument(
        '-d', '--train-data',
        required=True,
        type=str,
        help='Absolute path to training data.'
    )
    parser.add_argument(
        '-t', '--test-data',
        required=True,
        type=str,
        help='Absolute path to test data'
    )
    parser.add_argument(
        '-x', '--image-dir',
        default='images',
        type=str,
        help='images directory name.'
    )
    parser.add_argument(
        '-y', '--mask-dir',
        default='masks',
        type=str,
        help='masks directory name.'
    )
    parser.add_argument(
        '-o', '--model-output',
        required=True,
        type=str,
        help='Absolute path to for model checkpoint.'
    )
    parser.add_argument(
        '-b', '--batch-size',
        default=16,
        type=int,
        help='Mini-batch size.'
    )
    parser.add_argument(
        '-e', '--epochs',
        default=20,
        type=int,
        help='Number of epochs to train.'
    )
    parser.add_argument(
        '-l', '--learning-rate',
        default=1e-4,
        type=float,
        help='Learning rate for Adam optimizer.'
    )
    parser.add_argument("--num-classes", default=2, help="Number of classes",
                        type=int)
    parser.add_argument(
        '--layers-per-block',
        default="2,3,3",
        help="Number of layers in dense blocks."
    )
    parser.add_argument("--num_classes", default=2,
                        help="Number of classes",
                        type=int)
    parser.add_argument("--growth-k", default=16,
                        help="Growth rate for Tiramisu", type=int)
    return parser


def main(args):
    print(args)
    logger = logging.getLogger('Runner')
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    train_images_path = os.path.join(args.train_data, args.image_dir)
    if os.path.isdir(train_images_path) is False:
        err_str = 'Invalid training images data path: {}'.format(train_images_path)
        logger.error(err_str)
        raise Exception(err_str)

    train_masks_path = os.path.join(args.train_data, args.mask_dir)
    if os.path.isdir(train_images_path) is False:
        err_str = 'Invalid training masks data path: {}'.format(
            train_masks_path)
        logger.error(err_str)
        raise Exception(err_str)

    test_images_path = os.path.join(args.test_data, args.image_dir)
    if os.path.isdir(train_images_path) is False:
        err_str = 'Invalid test images data path: {}'.format(
            test_images_path)
        logger.error(err_str)
        raise Exception(err_str)

    test_masks_path = os.path.join(args.test_data, args.mask_dir)
    if os.path.isdir(train_images_path) is False:
        err_str = 'Invalid test masks data path: {}'.format(
            test_masks_path)
        logger.error(err_str)
        raise Exception(err_str)

    if os.path.isdir(args.model_output) is False:
        err_str = 'Invalid model output path: {}'.format(
            args.model_output)
        logger.error(err_str)
        raise Exception(err_str)

    layers_per_block = [int(x) for x in args.layers_per_block.split(",")]
    tiramisu = DenseTiramisu(args.growth_k, layers_per_block,
                             args.num_classes)

    tiramisu.train(args.train_data, args.test_data, args.model_output,
        args.batch_size, args.epochs, args.learning_rate)


if __name__ == "__main__":
    args = get_argparser().parse_args(sys.argv[1:])
    main(args)
