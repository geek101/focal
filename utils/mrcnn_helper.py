"""
Bablu
Helper methods for image and video processing for mrcnn inference.

Copyright (c) 2019 Powell Molleti.
Licensed under the MIT License (see LICENSE for details)

Taken and modified from:
https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/utils.py
"""

import numpy as np
import cv2
import datetime
import skimage
import tqdm
import collections
import os
import skimage.io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

import mrcnn.model as modellib
from mrcnn import utils as mrcnn_utils
from deps.coco import coco


def bounding_box_to_plt(image, b):
    """
    Convert one bounding box data into what mathplotlib understands
    [XMin1,    XMax1,     YMin1,   YMax1,        XMin2,    XMax2,    YMin2,   YMax2]
    ['0.005', '0.033125', '0.58', '0.62777776', '0.005', '0.033125', '0.58', '0.62777776']
    for: https://matplotlib.org/api/_as_gen/matplotlib.patches.Rectangle.html#matplotlib.patches.Rectangle
    """
    xsize = image.shape[1]
    ysize = image.shape[0]
    xy = (int(float(b[0]) * xsize), int(float(b[2]) * ysize))   # (XMin1 * xsize, YMin1 * ysize)
    width = int(float(b[1]) * xsize) - xy[0]        # XMax1 * xsize - XMin1 * xsize
    height = int(float(b[3]) * ysize) - xy[1]       # YMax1 * ysize - Ymin * ysize
    return (xy, width, height)


def two_bounding_boxes_to_plt(image, b):
    """
    Convert two bounding box data into what mathplotlib understands
    """
    return [bounding_box_to_plt(image, b[0:4]), bounding_box_to_plt(image,
                                                                    b[4:len(b)])]


def show_images(images, titles=None, bounding_boxes_list=[]):
    """Display a list of images"""
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    
    for i in range(0, len(images)):
        image = images[i]
        title = "None"
        if titles is not None and len(titles) > i:
            title = titles[i]
        
        bounding_boxes = None
        if bounding_boxes_list is not None and len(bounding_boxes_list) > i:
            bounding_boxes = bounding_boxes_list[i]

        a = fig.add_subplot(1,n_ims,n) # Make subplot
        if len(image.shape) == 2 or image.shape[2] == 1: # Is image grayscale?
            plt.imshow(np.resize(image, (image.shape[0], image.shape[1])),
                       interpolation="bicubic", cmap="gray")
        else:
            plt.imshow(image, interpolation="bicubic")
            if bounding_boxes is not None:
                box1, box2 = two_bounding_boxes_to_plt(image, bounding_boxes)
                rect1 = patches.Rectangle(
                    (box1[0]), box1[1], box1[2], linewidth=2,
                    edgecolor='y', facecolor='none')
                rect2 = patches.Rectangle(
                    (box2[0]), box2[1], box2[2], linewidth=2,
                    edgecolor='g', facecolor='none')
                a.add_patch(rect1)
                a.add_patch(rect2)
        if titles is not None:
            a.set_title(title + ' {}x{}'.format(image.shape[0], image.shape[1]))
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.axis('off')
    plt.show()


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


def get_id_to_name(class_names):
    class_id_to_name = {}
    for i in range(0, len(class_names)):
        class_id_to_name[i] = class_names[i]
    return class_id_to_name


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def get_mrcnn_model(model_path, log_path):
    config = InferenceConfig()
    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=log_path,
                              config=config)

    # Load weights trained on MS-COCO
    model.load_weights(model_path, by_name=True)

    return model


def run_mrcnn_single(image_path, model, verbose=0):
    image = skimage.io.imread(image_path)
    
    # Run detection
    results = model.detect([image], verbose=verbose)

    # get results
    r = results[0]
    return r


def get_mrcnn_class_count_dict(mrcnn_result, class_id_dict):
    class_name_to_count = collections.defaultdict(int)
    r = mrcnn_result
    boxes, masks, class_ids = r['rois'], r['masks'], r['class_ids']

    for i in range(boxes.shape[0]):
        class_name_to_count[class_id_dict[class_ids[i]]] += 1
        
    return class_name_to_count

def run_mrcnn(prefix, image_data_list, model_path, log_path):
    model = get_mrcnn_model(model_path, log_path)
    class_id_dict = get_id_to_name(CLASS_NAMES)
    output_dict = {}

    for image_path in image_data_list:
        image_abs_path = os.path.join(prefix, image_path)
        
        print('Processing image: {}'.format(image_abs_path))
        image = skimage.io.imread(image_abs_path)
        if len(image.shape) < 3 or image.shape[2] > 3:
            print('Skipping image: {}, shape: {}'.format(image_abs_path,
                                                         image.shape))
            continue

        mrcnn_result = run_mrcnn_single(image_abs_path, model)
        class_names_to_count = get_mrcnn_class_count_dict(mrcnn_result,
                                                          class_id_dict)

        output_dict[image_path] = class_names_to_count
    return output_dict


def apply_mask(image, mask, orig_image, trimap=False):
    """
    Apply the given mask to the image or use the original image.
    :param image: Given input image that is blurred.
    :param mask: Given mask where to apply original image.
    :param orig_image: Given original image.
    :return clobbered image as numpy object.
    """
    if trimap is True:
        image = np.where(mask == 1, 255, image)
    else:
        for c in range(3):
            image[:, :, c] = np.where(mask == 1, orig_image[:, :, c],
                                      image[:, :, c])
    return image


def display_instances_single(image, boxes, masks, class_ids, class_names,
                             show_mask=True, skip_class=17, trimap=False,
                             verbose=False):
    """
    Blur the portion of the image that is not classified as the skip class id.
    :param image: Input image to run inference on.
    :param boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image
    coordinates.
    :param masks: [height, width, num_instances]
    :param class_ids: [num_instances]
    :param class_names: list of class names of the dataset
    :param show_mask: To show masks or not
    :param skip_class: Which class id to mask
    :param verbose: Print extra debug info
    :return: numpy image where everything other than the class is blurred.
    """

    # Number of instances
    N = boxes.shape[0]
    if verbose is True:
        if not N:
            print("\n*** No instances to display *** \n")
        else:
            print('Boxes shape: {}, masks shape: {}, class_ids.shape: {}'.
                  format(boxes.shape, masks.shape, class_ids.shape))
            assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    if trimap is False:
        masked_image = cv2.blur(image,(int(image.shape[0]/4),
                                       int(image.shape[0]/4)))
    else:
        masked_image = np.zeros((image.shape[0], image.shape[1]))

    max_square = None
    max_mask = None
    for i in range(N):
        if class_ids[i] == skip_class:
            # compute the square of each object
            y1, x1, y2, x2 = boxes[i]
            square = abs(y2 - y1) * abs(x2 - x1)

            if verbose is True:
                print("Processing class: {}-{}".format(class_ids[i],
                                                       class_names[class_ids[i]]))

            if max_square is None or max_square < square:
                mask = masks[:, :, i]
                max_square = square
                max_mask = mask
                continue

    if show_mask and max_mask is not None:
        masked_image = apply_mask(masked_image, max_mask, image, trimap=trimap)

    return masked_image


def detect_and_mask(model, class_names, skip_class=17, image_path=None,
                    video_path=None):
    """
    Helper for both image and video that wraps the display_instances_single.
    :param model: Given model to run inference on
    :param class_names: names of all classes
    :param skip_class: Class id to not blur
    :param image_path: path of the image file
    :param video_path: path of the video file
    :param verbose: Enable/Disable debug logging
    :return: Output file path
    """
    assert image_path or video_path

    # Image or video?
    file_name = None
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(image_path))
        # Read image
        image = skimage.io.imread(image_path)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        masked_image = display_instances_single(image, r['rois'], r['masks'], r['class_ids'],
                                        class_names, skip_class=skip_class)
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(
            datetime.datetime.now())
        skimage.io.imsave(file_name, masked_image)
    elif video_path:
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        length = int(vcapture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.mp4".format(
            datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MP4V'),
                                  fps, (width, height))

        success = True
        pbar = tqdm.tqdm(total=length)
        while success:
            pbar.update(1)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                resized_image=cv2.resize(image, (600, 800))
                # Detect objects
                r = model.detect([resized_image], verbose=0)[0]
                # Color splash
                masked_image = display_instances_single(
                    resized_image, r['rois'], r['masks'], r['class_ids'],
                    class_names, skip_class=skip_class)
                # RGB -> BGR to save image to video
                masked_image = masked_image[..., ::-1]
                masked_image = masked_image = cv2.resize(
                    masked_image, (image.shape[1], image.shape[0]))
                # Add image to video writer
                vwriter.write(masked_image)
        vwriter.release()
        # print("Saved to ", file_name)
    return file_name, length


def resize_image(image, min_dim=None, max_dim=None, min_scale=None,
                 mode="square"):
    """Resizes an image keeping the aspect ratio unchanged.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    min_scale: if provided, ensure that the image is scaled up by at least
        this percent even if min_dim doesn't require it.
    mode: Resizing mode.
        none: No resizing. Return the image unchanged.
        square: Resize and pad with zeros to get a square image
            of size [max_dim, max_dim].
        pad64: Pads width and height with zeros to make them multiples of 64.
               If min_dim or min_scale are provided, it scales the image up
               before padding. max_dim is ignored in this mode.
               The multiple of 64 is needed to ensure smooth scaling of feature
               maps up and down the 6 levels of the FPN pyramid (2**6=64).
        crop: Picks random crops from the image. First, scales the image based
              on min_dim and min_scale, then picks a random crop of
              size min_dim x min_dim. Can be used in training only.
              max_dim is not used in this mode.

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        image = mrcnn_utils.resize(image, (round(h * scale), round(w * scale)),
                                   preserve_range=True)

    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        # XXX: Modified when compared to what exists in MRCNN, to deal
        # with gray scale images.
        padding = padding[:len(image.shape)]

        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        # XXX: Modified when compared to what exists in MRCNN, to deal
        # with gray scale images.
        padding = padding[:len(image.shape)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), window, scale, padding, crop
