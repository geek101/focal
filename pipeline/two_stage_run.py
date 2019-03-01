"""
Copyright (c) 2019 Powell Molleti.
Licensed under the MIT License (see LICENSE for details)

CLI for two_stage_process.py
"""

import argparse
import os
import logging
import sys

from pipeline.deeplabv3_inference import DeepLabInference
from pipeline.mrcnn_inference import MRCNNInference
from pipeline.two_stage_process import ProcessSM


def get_argparser():
    parser = argparse.ArgumentParser(description='Mask_RCNN Runner')
    parser.add_argument(
        '-m', '--mrcnn-model-path',
        required=True,
        type=str,
        help='Absolute path to mask rcnn model file.'
    )
    parser.add_argument(
        '-l', '--mrcnn-log-dir',
        required=True,
        type=str,
        help='Absolute path to mask rcnn log folder.'
    )
    parser.add_argument(
        '-d', '--deeplab-model-path',
        required=True,
        type=str,
        help='Absolute path to mask rcnn log folder.'
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
    print(args)
    logger = logging.getLogger('Runner')
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    if os.path.isfile(args.mrcnn_model_path) is False:
        err_str = 'Invalid Mask RCNN model path: {}'.format(
            args.mrcnn_model_path)
        logger.error(err_str)
        raise Exception(err_str)

    if os.path.isfile(args.deeplab_model_path) is False:
        err_str = 'Invalid DeepLab model path: {}'.format(
            args.args.deeplab_model_path)
        logger.error(err_str)
        raise Exception(err_str)

    mask_rcnn_model_config = {'gpu_count': 1, 'images_per_gpu': 1,
                              'log_dir': os.path.abspath(args.mrcnn_log_dir),
                              'model_path': os.path.abspath(
                                  args.mrcnn_model_path)}

    if os.path.isfile(args.input_video) is False:
        err_str = 'Invalid input video: {}'.format(
            args.args.input_video)
        logger.error(err_str)
        raise Exception(err_str)

    deeplab_obj = DeepLabInference(
        tarball_path=os.path.abspath(args.deeplab_model_path))
    maskrcnn_obj = MRCNNInference(mask_rcnn_model_config)

    process_sm = ProcessSM(maskrcnn_obj, deeplab_obj)

    process_sm.run_helper(args, logger)


if __name__ == "__main__":
    args = get_argparser().parse_args(sys.argv[1:])
    main(args)    
