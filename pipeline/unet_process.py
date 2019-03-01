"""
Focal                                                                         
Unet process API                                                         
Copyright (c) 2019 Powell Molleti.                                             
Licensed under the MIT License (see LICENSE for details)                      
"""

import cv2
import logging
import numpy as np

from pipeline.process_base import ProcessBase


class UNetProcess(ProcessBase):
    def __init__(self, unet_infer_obj):
        self.logger = logging.getLogger('UNetProcess')
        self._unet_infer = unet_infer_obj

        self._frame_processed_count = 0

    def run_image(self, image):
        """
        Expects an RGB image.
        Output is a masked image.
        """
        mask = self._unet_infer.run(image)

        output_image = cv2.blur(image,
                                (int(image.shape[0]/4),
                                 int(image.shape[1]/4)))
        for i in range(0, 3):
            output_image[:, :, i] = np.where(mask == 1,
                                             image[:, :, i],
                                             output_image[:, :, i])

        return output_image
