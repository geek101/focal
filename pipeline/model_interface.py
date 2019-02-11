"""
Copyright (c) 2019 Powell Molleti.
Licensed under the MIT License (see LICENSE for details)
"""


from abc import ABCMeta, abstractmethod


class ModelInterface(metaclass=ABCMeta):
    @abstractmethod
    def run(self, input_images, model_config):
        raise NotImplementedError("Please Implement this method")
