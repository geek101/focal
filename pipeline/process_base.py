"""                                                                            
Focal                                                                          
Unet process API                                                               
Copyright (c) 2019 Powell Molleti.                                             
Licensed under the MIT License (see LICENSE for details)
"""

import cv2
import os
import tqdm
import time

from abc import ABCMeta, abstractmethod


class ProcessBase(metaclass=ABCMeta):
    @abstractmethod
    def run_image(self, image):
        raise NotImplementedError("Please Implement this method")

    def run(self, video_input_file, video_output_file):
        """                                                                    
        Video processor, does it frame by frame.                               
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
        return video_output_file, length

    def run_helper(self, args, logger):
        logger.info('Processing input video: {}'.format(
            os.path.abspath(args.input_video)))
        start_time = time.time()
        output_file, count = self.run(
            os.path.abspath(args.input_video),
            os.path.abspath(args.output_video))
        end_time = time.time()
        diff_time = 0
        if end_time > start_time:
            diff_time = end_time - start_time
        fps = 0
        if diff_time > 0:
            fps = int(round(float(count) / diff_time, 0))
        logger.info('Frame count: {}, fps: {}'.format(count, fps))

