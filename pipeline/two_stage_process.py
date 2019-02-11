"""
Copyright (c) 2019 Powell Molleti.
Licensed under the MIT License (see LICENSE for details)

API that processs the input raw frames using two stages,
i.e Mask RCNN and then DeepLabV3
"""

import logging
import cv2
import tqdm
import numpy as np

class ProcessSM(object):
    def __init__(self, mask_rcnn_obj, deeplab_obj,
                 threshold=0.90,
                 filter_class='person'):
        self.logger = logging.getLogger('ProcessSM')
        self._mask_rcnn_obj = mask_rcnn_obj
        self._deeplab_obj = deeplab_obj

        self._threshold = threshold
        self._filter_class = filter_class
        self._model_config = {}
        self._model_config['filter_class'] = self._filter_class
        self._max_mask = None
        self._frame_refresh = 15
        self._frame_processed_count = 0

    def mask_iou_calc(self, target, prediction, target_class, prediction_class):
        target = target == target_class
        prediction = prediction == prediction_class
        target = target.astype('uint8')
        prediction = prediction.astype('uint8')
        intersection = np.logical_and(target, prediction)
        union = np.logical_or(target, prediction)
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score
        
    def run_image(self, image):
        if self._max_mask is None:
            r = self._mask_rcnn_obj.run(image, self._model_config)
            self._max_mask = self._mask_rcnn_obj.get_max_mask(
                image, r, self._model_config['filter_class'])

            if self._max_mask is None:
                return image

            self._second_image = np.zeros((image.shape[0], image.shape[1],
                                           image.shape[2])).astype('uint8')
            for c in range(3):
                self._second_image[:, :, c] = np.where(
                    self._max_mask == self._mask_rcnn_obj.get_class_id(
                        self._filter_class),
                    image[:, :, c], self._second_image[:, :, c])

        image_out, seg_map = self._deeplab_obj.run(self._second_image,
                                                   self._model_config)
        
        masked_image = cv2.blur(image, (int(image.shape[0]/4),
                                        int(image.shape[1]/4)))

        for c in range(3):
            masked_image[:, :, c] = np.where(
                seg_map == self._deeplab_obj.get_class_id(self._filter_class),
                image[:, :, c], masked_image[:, :, c])

        # Calculate IOU and reset max_mask if necessary
        mask_iou = self.mask_iou_calc(self._max_mask, seg_map,
                      self._mask_rcnn_obj.get_class_id(self._filter_class),
                      self._deeplab_obj.get_class_id(self._filter_class))

        if mask_iou < self._threshold:
            self._max_mask = None
            
        self._frame_processed_count += 1
        if self._frame_processed_count % self._frame_refresh == 0:
            self._max_mask = None
            
        return masked_image

    def run(self, video_input_file, video_output_file):
        """
        State machine that gets the max human mask using Mask RCNN then
        rest of the frames are processed with DeeplabV3.
        Runs Mask RCNN again if DeeplabV3 intersection with mask decreases
        by a threadhold.
        """

        # Video capture
        vcapture = cv2.VideoCapture(video_input_file)
        length = int(vcapture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        vwriter = cv2.VideoWriter(video_output_file,
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
                masked_image = self.run_image(image)
                # RGB -> BGR to save image to video
                masked_image = masked_image[..., ::-1]
                # Add image to video writer
                vwriter.write(masked_image)
        vwriter.release()
        # print("Saved to ", file_name)
        return video_output_file, length
