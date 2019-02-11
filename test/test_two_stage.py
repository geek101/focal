"""
Copyright (c) 2019 Powell Molleti.
Licensed under the MIT License (see LICENSE for details)

Unit test for two stage process.
"""

import unittest
import tempfile
import skimage.io
import numpy as np

from pipeline.deeplabv3_inference import DeepLabInference
from pipeline.mrcnn_inference import MRCNNInference
from pipeline.two_stage_process import ProcessSM

class TwoStageProcessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deeplab_path = \
            'pretrained_models/deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz'
        cls.maskrcnn_path = 'pretrained_models/mask_rcnn_coco.h5'
        cls.mrcnn_log_dir = tempfile.mkdtemp()
        cls.mask_rcnn_model_config = {'gpu_count': 1, 'images_per_gpu': 1, 
                                      'log_dir': cls.mrcnn_log_dir,
                                      'model_path': cls.maskrcnn_path}
        cls.deeplab_obj = DeepLabInference(tarball_path=cls.deeplab_path)
        cls.maskrcnn_obj = MRCNNInference(cls.mask_rcnn_model_config)
        cls.orig_image = skimage.io.imread('data/images/3_carena_image.jpeg')
        cls.target_image = np.asarray(skimage.io.imread(
            'data/images/3_carena_image_masked.png'))

    def test_smoke(self):
        process_sm = ProcessSM(self.maskrcnn_obj, self.deeplab_obj)
        masked_image = process_sm.run_image(image=self.orig_image)
        self.assertTrue(np.array_equal(masked_image, self.target_image))
        
if __name__ == '__main__':
    unittest.main()
