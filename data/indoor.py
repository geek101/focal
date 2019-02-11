"""
Bablu
Indoor data set mrcnn processing helper.

Copyright (c) 2019 Powell Molleti.
Licensed under the MIT License (see LICENSE for details)

dataset: http://web.mit.edu/torralba/www/indoor.html
"""

import glob
import os

def get_image_paths(prefix, suffix='jpg'):
    images_dir=os.path.join(prefix, '*/*.{}'.format(suffix))
    images_paths = glob.glob(images_dir)
    return [ '/'.join(x.split('/')[-2:]) for x in images_paths ]
