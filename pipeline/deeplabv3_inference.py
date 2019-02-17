"""
Copyright (c) 2019 Powell Molleti.
Licensed under the MIT License (see LICENSE for details)

https://github.com/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb
"""

import os
import logging

import tarfile
import tempfile
from six.moves import urllib

import numpy as np
from PIL import Image
import tensorflow as tf

from pipeline.model_interface import ModelInterface


class DeepLabInference(ModelInterface):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    # @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval'
    MODEL_NAME = 'mobilenetv2_coco_voctrainaug'

    _DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
    _MODEL_URLS = {
        'mobilenetv2_coco_voctrainaug':
            'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
    }
    _TARBALL_NAME = 'deeplab_model.tar.gz'

    LABEL_NAMES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
    ]

    @staticmethod
    def _download():
        model_dir = tempfile.mkdtemp()
        tf.gfile.MakeDirs(model_dir)

        download_path = os.path.join(model_dir, DeepLabInference._TARBALL_NAME)
        print('downloading model, this might take a while...')
        urllib.request.urlretrieve(DeepLabInference._DOWNLOAD_URL_PREFIX +
                                   DeepLabInference._MODEL_URLS[
                                       DeepLabInference.MODEL_NAME],
                                   download_path)
        return download_path

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.logger = logging.getLogger('DeepLabInferencel')
        self._class_idx_dict = {}
        for i in range(0, len(DeepLabInference.LABEL_NAMES)):
            self._class_idx_dict[DeepLabInference.LABEL_NAMES[i]] = i

        self.graph = tf.Graph()

        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def get_class_id(self, class_name):
        return self._class_idx_dict[class_name]

    def run(self, input_image, model_config):
        """Runs inference on a single image.

        Args:
          image: A PIL.Image object, raw input image.

        Returns:
          resized_image: RGB image resized from original input image.
          seg_map: Segmentation map of `resized_image`.
        """
        image = Image.fromarray(input_image.astype('uint8'))
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size,
                                                    Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]

        seg_map = Image.fromarray(seg_map.astype('uint8')).resize(
            (input_image.shape[1], input_image.shape[0]), Image.ANTIALIAS)

        return input_image, np.asarray(seg_map)
