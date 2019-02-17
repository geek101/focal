"""
Copyright (c) 2019 Powell Molleti.
Licensed under the MIT License (see LICENSE for details)
"""

import os
import logging

import mrcnn.model as modellib
from pipeline.coco import coco

from pipeline.model_interface import ModelInterface


class MRCNNInference(ModelInterface):
    """
    Model initialization and inference runner.
    """
    def __init__(self, model_config):
        self.logger = logging.getLogger('MRCNNInference')
        self._gpu_count = int(model_config['gpu_count'])
        self._images_per_gpu = int(model_config['images_per_gpu'])

        class InferenceConfig(coco.CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = int(model_config['gpu_count'])
            IMAGES_PER_GPU = int(model_config['images_per_gpu'])

        self._config = InferenceConfig()
        self._model_dir = model_config['log_dir']
        self._pretrained_model_path = model_config['model_path']

        self._class_names = [
               'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
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

        self._class_idx_dict = {}
        for i in range(0, len(self._class_names)):
            self._class_idx_dict[self._class_names[i]] = i

        if os.path.isdir(self._model_dir) is False:
            str_error = "Valid model dir is required, given dir: {}" \
                        " is not found".format(self._model_dir)
            self.logger.error(str_error)
            raise Exception(str_error)

        if os.path.isfile(self._pretrained_model_path) is False:
            str_error = "Valid pre-trained model path is required, "\
                        "given path: {} is not found".format(
                            self._pretrained_model_path)
            self.logger.error(str_error)
            raise Exception(str_error)

        # Create model object in inference mode.
        self._model = modellib.MaskRCNN(
            mode="inference", model_dir=self._model_dir, config=self._config)

        # Load weights trained on MS-COCO
        self._model.load_weights(self._pretrained_model_path, by_name=True)

    def get_class_id(self, class_name):
        return self._class_idx_dict[class_name]
    
    def get_max_mask(self, result, filter_class):
        boxes, masks, class_ids = result['rois'], result['masks'], \
                                  result['class_ids']
        max_square = None
        max_mask = None
        for i in range(boxes.shape[0]):
            if class_ids[i] == self._class_idx_dict[filter_class]:
                # compute the square of each object
                y1, x1, y2, x2 = boxes[i]
                square = abs(y2 - y1) * abs(x2 - x1)

                if max_square is None or max_square < square:
                    mask = masks[:, :, i]
                    max_square = square
                    max_mask = mask
                    continue
        return max_mask
            
    def run(self, input_image, model_config):
        results = self._model.detect([input_image])
        r = results[0]
        return r

