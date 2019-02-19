"""
Bablu
Runner for Mask RCNN on given data.

Copyright (c) 2019 Powell Molleti.
Licensed under the MIT License (see LICENSE for details)
"""

import argparse
import pickle
import sys

from data.indoor import get_image_paths
from utils.mrcnn_helper import run_mrcnn


def get_argparser():
    parser = argparse.ArgumentParser(description='Mask_RCNN Runner')
    parser.add_argument(
        '--model-path',
        type=str,
        help='Absolute path to model file.'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        help='Absolute path to log older.'
    )
    parser.add_argument(
        '--data-prefix',
        type=str,
        help='Absolute prefix to data directory.'
    )
    parser.add_argument(
        '--pickle-file',
        type=str,
        help='Absolute path to pickle output.'
    )
    return parser


def main(args_input):
    prefix = args_input.data_prefix
    image_data_list = get_image_paths(prefix)
    assert len(image_data_list) != 0
    print('prefix: {}, length of image list: {}'.format(
        prefix, len(image_data_list)))
    model_path = args_input.model_path
    log_dir = args.log_dir
    output_dict = run_mrcnn(prefix, image_data_list, model_path, log_dir)

    with open(args_input.pickle_file, 'wb') as f:
        pickle.dump(output_dict, f)
        
    print('Pickle file: {}'.format(args_input.pickle_file))


if __name__ == "__main__":
    args = get_argparser().parse_args(sys.argv[1:])
    main(args)    
