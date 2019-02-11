#!/bin/bash

if [ ! -d 'Mask_RCNN' ]; then
    git clone https://github.com/matterport/Mask_RCNN.git
fi

cd Mask_RCNN
pip3 install -r requirements.txt
python3 setup.py install
cd -

pip3 uninstall --yes tensorflow
pip3 uninstall --yes Keras

pip3 install -r requirements.txt

if [ ! -d 'pretrained_models' ]; then
    mkdir -p pretrained_models
fi

cd pretrained_models
if [ ! -f 'mask_rcnn_coco.h5' ]; then
    wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
fi

if [ ! -f 'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz' ]; then
    wget http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz
fi

cd -

