"""
Bablu
Combine portrait images and indoor dataset with augmentation.

Copyright (c) 2019 Powell Molleti.
Licensed under the MIT License (see LICENSE for details)

dataset: http://web.mit.edu/torralba/www/indoor.html
"""

import random
import scipy.ndimage
import skimage.color
import numpy as np

from skimage.transform import warp, AffineTransform
from utils.mrcnn_helper import resize_image


def warp_helper(img, affine_t):
    return warp(img, affine_t, mode='constant', cval=0.0,
                preserve_range=True).astype(np.uint8)


def random_affine_helper(img, img_mask, intensity=1.0, rotation_disabled=True,
                         shear_disabled=True, scale_disabled=True):
    if rotation_disabled:
        rotation = None
    else:
        rotation = random.uniform(-.15 * intensity, .15 * intensity)

    if shear_disabled:
        shear = None
    else:
        shear = random.uniform(-.15 * intensity, .15 * intensity)

    if scale_disabled:
        scale = None
    else:
        scale_rnd = random.uniform(.9, 1.1)
        scale = (scale_rnd, scale_rnd)

    affine_t = AffineTransform(rotation=rotation, shear=shear, scale=scale)
    return warp_helper(img, affine_t), warp_helper(img_mask, affine_t)


def shift_helper(image, shift):
    return scipy.ndimage.shift(image, shift=shift, mode='constant', cval=0)


def shift_mask_corners(portrait_image, portrait_mask, shift_arg='none'):
    """
    Given resize portrait image and the mask shift the mask
    and the image to right and left corner. With some randomness
    to enable partial occlusion of left or right side of the body
    and or face.
    shift_arg options:
    ['none', 'left_corner', 'right_corner', 'left_random', 'right_random']
    """
    aw = np.argwhere(portrait_mask != 0)

    aw_col1 = aw[:, 1:]
    # print(aw_col1.reshape(aw_col1.shape[0]))
    # print('Min: {}'.format(aw_col1.min()))
    # print('Max: {}'.format(aw_col1.max()))

    shift_param = shift_arg.strip().lower()
    col1_min = aw_col1.min()
    if shift_param == 'left_corner':
        shift = [0, -col1_min, 0]
    elif shift_param == 'right_corner':
        shift = [0, col1_min, 0]
    elif shift_param == 'left_random':
        shift = [0, random.randint(int(-col1_min / 2), int(-col1_min / 4)), 0]
    elif shift_param == 'right_random':
        shift = [0, random.randint(int(col1_min / 4), int(col1_min / 2)), 0]
    elif shift_param == 'none':
        shift = [0, 0, 0]
    else:
        raise Exception('Invalid shift arg: {}, allow params: {}'.format(
            shift_arg, ['none', 'left_corner', 'right_corner', 'left_random',
                        'right_random']))

    return shift_helper(portrait_image, shift), shift_helper(portrait_mask,
                                                             shift[:2])


def convert_to_color_safe(input_image):
    if len(input_image.shape) == 2 or input_image.shape[2] == 1:
        return skimage.color.grey2rgb(input_image).astype(dtype=np.uint8)
    else:
        return input_image.astype(dtype=np.uint8)


def diff_pad(diff):
    flip = random.randint(0, 1) == 1
    pad = (0, 0)
    if diff > 0:
        if flip:
            pad = (diff - int(diff / 2), int(diff / 2))
        else:
            pad = (int(diff / 2), diff - int(diff / 2))

    return pad


def embed_helper(extra_padded, extra_padded_mask, target_image_input):
    target_image = convert_to_color_safe(target_image_input)
    pasted_image = np.zeros(target_image.shape, dtype='uint8')

    for c in range(3):
        pasted_image[:, :, c] = np.where(extra_padded_mask == 1,
                                         extra_padded[:, :, c],
                                         target_image[:, :, c])

    return pasted_image, extra_padded_mask


def portrait_with_mask_resize_to_indoor(portrait_image_input,
                                        portrait_mask_input,
                                        indoor_image_input):
    """
    Resize the portrait to indoor image size, keeping the aspect ratio.
    Pad with zeros.
    Returns resized image with padding, resized maskwith padding
    """
    portrait_image = convert_to_color_safe(portrait_image_input)

    indoor_height, indoor_width = indoor_image_input.shape[:2]
    portrait_height, portrait_width = portrait_image_input.shape[:2]

    portrait_resized = portrait_image
    portrait_mask_resized = portrait_mask_input
    if portrait_height > indoor_height or portrait_width > indoor_width:
        height_diff = portrait_height - indoor_height
        width_diff = portrait_width - indoor_width
        resize_dim = indoor_height
        if width_diff > height_diff:
            resize_dim = indoor_width

        portrait_resized = \
        resize_image(portrait_image, min_dim=resize_dim, max_dim=resize_dim,
                     mode="square")[0]
        portrait_mask_resized = \
        resize_image(portrait_mask_input, min_dim=resize_dim,
                     max_dim=resize_dim, mode="square")[0]

    portrait_resized_height, portrait_resized_width = portrait_resized.shape[:2]
    h_diff = indoor_height - portrait_resized_height
    w_diff = indoor_width - portrait_resized_width

    h_pad = (0, 0)
    if h_diff > 0:
        h_pad = (h_diff, 0)

    w_pad = diff_pad(w_diff)

    npad = (h_pad, w_pad, (0, 0))
    extra_padded = np.pad(portrait_resized, pad_width=npad, mode='constant',
                          constant_values=0)
    extra_padded_mask = np.pad(portrait_mask_resized, pad_width=npad[:2],
                               mode='constant', constant_values=0)

    return extra_padded, extra_padded_mask


def portrait_indoor_embed(portrait_image_input, portrait_mask_input,
                          indoor_image_input, shift_arg='none',
                          random_affine=False, intensity=1.0):
    """
    Helper to embed portrait mask section into the indoor
    image with augmentation.
    """

    extra_padded, extra_padded_mask = portrait_with_mask_resize_to_indoor(
        portrait_image_input, portrait_mask_input, indoor_image_input)

    extra_padded, extra_padded_mask = shift_mask_corners(extra_padded,
                                                         extra_padded_mask,
                                                         shift_arg=shift_arg)
    if shift_arg.strip().lower() != 'none' and random_affine:
        extra_padded, extra_padded_mask = random_affine_helper(
            extra_padded, extra_padded_mask, intensity=intensity,
            shear_disabled=False, scale_disabled=False)

    return embed_helper(extra_padded, extra_padded_mask, indoor_image_input)
