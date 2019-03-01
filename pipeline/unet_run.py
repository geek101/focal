"""
Focal
Unet runner

Copyright (c) 2019 Powell Molleti.
Licensed under the MIT License (see LICENSE for details)
"""

import argparse
import logging
import sys


from pipeline.unet_inference import UNetInference
from pipeline.unet_process import UNetProcess


def get_argparser():
    parser = argparse.ArgumentParser(description='UNet Runner')
    parser.add_argument(
        '-m', '--model-path',
        required=True,
        type=str,
        help='Absolute path to unet keras model file.'
    )
    parser.add_argument(
        '-r', '--resize',
        default=256,
        type=int,
        help='Image resize size.'
    )
    parser.add_argument(
        '-i', '--input-video',
        required=True,
        type=str,
        help='Absolute path to input video.'
    )
    parser.add_argument(
        '-o', '--output-video',
        required=True,
        type=str,
        help='Absolute path to file for video output.'
    )
    return parser


def main(args):
    logger = logging.getLogger('UNet Runner')
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    unet_infer = UNetInference(args.model_path, image_resize=args.resize)
    unet_process = UNetProcess(unet_infer)

    unet_process.run_helper(args, logger)


if __name__ == "__main__":
    args = get_argparser().parse_args(sys.argv[1:])
    main(args)
