"""
Focal
Unet inference runner

poLicensed under the MIT License (see LICENSE for details)
"""

import cv2
import os
import logging
import numpy as np

from keras.models import load_model

from utils.mrcnn_helper import resize_image
from pipeline.unet_model import _mean_iou_unet_wrapper

from pipeline.model_interface import ModelInterface


class UNetInference(ModelInterface):
    """Class to load UNet model and run inference."""

    def __init__(self, model_path, image_resize=256):
        self.logger = logging.getLogger('UNetInference')

        self._model_path = os.path.abspath(model_path)
        if os.path.isfile(self._model_path) is False:
            err_str = 'Invalid model path provided: {}'.format(
                self._model_path)
            self.logger.error(err_str)
            raise Exception(err_str)

        self._unet_infer = load_model(
            self._model_path,
            custom_objects={'_mean_iou_unet_wrapper': _mean_iou_unet_wrapper })

        self._resize = image_resize


    def _normalize_image(self, image_orig):
        return image_orig/255
    
    def _process_input_image(self, image_orig):
        """
        Normalize the image, convert to grayscale and resize.
        Return the processed image
        """
        image = image_orig
        if len(image.shape) > 2 and image.shape[2] > 1:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        image = self._normalize_image(image)
        
        image_resized, window, scale, padding, crop = resize_image(
            image, min_dim=self._resize, max_dim=self._resize, mode="square")

        if len(image_resized.shape) < 3:
            image_resized = image_resized.reshape(
                (image_resized.shape[0],image_resized.shape[1], 1))

        assert image_resized.shape == (256, 256, 1)

        return image_resized, window

    def run(self, input_image, model_config=None):
        # Prepare the input image, get the window into it.
        image_processed, window = self._process_input_image(input_image)
        
        image_mask = self._unet_infer.predict(
            np.array([image_processed]))[0]

        # Convert to binary mask
        image_mask = (image_mask > 0.5).astype('uint8')
        
        # Upscale the mask output
        w = window
        mask_window = image_mask[w[0]:w[2], w[1]:w[3], :]
        mask_window_orig = cv2.resize(mask_window,
                                      (input_image.shape[1],
                                       input_image.shape[0]))
        return mask_window_orig
