## Focal - Auto privacy for video conferencing.

![alt text][vinv]

```* (original video)``` [https://www.youtube.com/watch?v=Mh4f9AYRCZY&feature=youtu.be](https://www.youtube.com/watch?v=Mh4f9AYRCZY&feature=youtu.be)

## Install

Run `$ sh install_deps.sh` after setting up a virtualenv.
Run the test case `$ python test/test_two_stage.py` to verify setup.

## Two stage model experiment

### Mask RCNN stage 1 pipeline

Mask RCNN is only run once every 15 frames or when the IOU does not reach 90%
between stage 1 and stage 2 model output. This model is pre-trained on COCO
dataset.

![alt text][mask]

### Mobilenetv2 with Deeplabv3+ stage 2 pipeline

This model uses the mask from stage 1 and refines it. Pretrained on
Pascal VOC 2012

### How to use

`$ python pipeline/two_stage_run.py -m pretrained_models/mask_rcnn_coco.h5 -l /tmp/logs -d pretrained_models/deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz -i data/videos/bbc_video_youtube.mp4 -o data/videos/processed_bbc_video_youtube.mp4`

### Performance

On NVidia 1080Ti the above video is typically processed at 18fps.

## Single stage model experiment(s)

### UNet

#### Input/Output

Input is grayscale image scaled to 256x256, output is mask of 256x256

#### Performance

![alt text][unet]

25 to 35 fps on 1080Ti.

#### How to use

Download [unet_focal-05-0.03.hdf5](https://github.com/geek101/focal/releases/download/v0.1-alpha/unet_focal-05-0.03.hdf5).

`$ python pipeline/unet_run.py --model-path pretrained_models/unet_model/unet_focal-05-0.03.hdf5 --input-video data/videos/bbc_news_clip_test.mp4 --output-video data/videos/unet_processed_bbc_news_clip_test.mp4`

## TODO

1. Improve performance.
2. Continue experiments to replace Mask RCNN usage.

## Citation

*   Mask RCNN 
```
@misc{matterport_maskrcnn_2017,
  title={Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow},
  author={Waleed Abdulla},
  year={2017},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/matterport/Mask_RCNN}},
}
```
*   DeepLabv3+:
```
@inproceedings{deeplabv3plus2018,
  title={Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation},
  author={Liang-Chieh Chen and Yukun Zhu and George Papandreou and Florian Schroff and Hartwig Adam},
  booktitle={ECCV},
  year={2018}
}
```


*   MobileNetv2:
```
@inproceedings{mobilenetv22018,
  title={MobileNetV2: Inverted Residuals and Linear Bottlenecks},
  author={Mark Sandler and Andrew Howard and Menglong Zhu and Andrey Zhmoginov and Liang-Chieh Chen},
  booktitle={CVPR},
  year={2018}
}
```

*   Deep Automatic Portrait Matting
```
@inproceedings{
  title={Deep Automatic Portrait Matting},
  author={Xiaoyong Shen, Xin Tao, Hongyun Gao, Chao Zhou, Jiaya Jia},
  booktitle={ECCV},
  year={2016}
}
```
*   U-Net: Convolutional Networks for Biomedical Image Segmentation
```
@inproceedings{
  title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author={Olaf Ronneberger, Philipp Fischer, Thomas Brox},
  booktitle={MICCAI},
  year={2015}
}
```

[unet]: https://github.com/geek101/focal/blob/master/data/readme/unet_processed_bbc_news_clip_test.gif "UNet bbc video processed"
[mask]: https://github.com/geek101/focal/blob/master/data/readme/mask_rcnn_pipeline.jpeg "Stage1 Inference output"

[vinv]: https://github.com/geek101/focal/blob/master/data/videos/bbc_clip_processed_vinv.gif "bbc video processed"
