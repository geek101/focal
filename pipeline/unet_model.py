"""
Focal
Keras implementation of Unet:
https://github.com/zhixuhao/unet

Copyright (c) 2019 Powell Molleti.
Licensed under the MIT License (see LICENSE for details)
"""

from keras.layers import Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import UpSampling2D, concatenate, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf


# https://github.com/Golbstein/KerasExtras/blob/master/keras_functions.py
def sparse_Mean_IOU(y_true, y_pred, nb_classes=2):
    iou = []
    pred_pixels = K.argmax(y_pred, axis=-1)
    for i in range(0, nb_classes):
        true_labels = K.equal(y_true[:,:,0], i)
        pred_labels = K.equal(pred_pixels, i)
        inter = tf.to_int32(true_labels & pred_labels)
        union = tf.to_int32(true_labels | pred_labels)
        legal_batches = K.sum(tf.to_int32(true_labels), axis=1)>0
        ious = K.sum(inter, axis=1)/K.sum(union, axis=1)
        iou.append(K.mean(tf.gather(ious, indices=tf.where(legal_batches)))) # returns average IoU of the same objects
    iou = tf.stack(iou)
    legal_labels = ~tf.debugging.is_nan(iou)
    iou = tf.gather(iou, indices=tf.where(legal_labels))
    return K.mean(iou)

def _simple_iou(y_true, y_pred, nb_classes=2):
    """
    Expects tensor of 4 dimensions with last dimension is just 1.
    Perfect for binary classification i.e no one hot vector encoding
    of last dimension.
    """
    iou = []
    for i in range(0, nb_classes):
        true_labels = K.equal(y_true, i)
        pred_labels = K.equal(y_pred, i)
        union = tf.to_int32(pred_labels | true_labels)
        inter = tf.to_int32(pred_labels & true_labels)
    
        union_sum = K.sum(K.sum(union, axis=1), axis=1)
        inter_sum = K.sum(K.sum(inter, axis=1), axis=1)
        ious = inter_sum / union_sum
        legal_batches = K.sum(K.sum(tf.to_int32(true_labels),
                                    axis=1), axis=1)>0
        iou.append(K.mean(tf.gather(ious,
                                    indices=tf.where(legal_batches))))
    iou = tf.stack(iou)
    legal_labels = ~tf.debugging.is_nan(iou)
    iou = tf.gather(iou, indices=tf.where(legal_labels))
    return K.mean(iou)


def _mean_iou_unet_wrapper(y_true, y_pred):
    y_pred_thresh = tf.to_int32(y_pred > 0.5)

    y_true_reshaped = tf.reshape(y_true,
                                 tf.shape(y_true)[:-1])
                                 
    y_pred_reshaped = tf.reshape(y_pred_thresh,
                                 tf.shape(y_pred_thresh)[:-1])
    return _simple_iou(y_true_reshaped, y_pred_reshaped)

def unet(pretrained_weights=None, input_size=(256, 256, 1),
         nb_classes=2, use_bn=True, kernel_initializer='he_uniform'):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer=kernel_initializer)(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer=kernel_initializer)(conv1)
    conv1 = BatchNormalization()(conv1) if use_bn else conv1
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer=kernel_initializer)(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer=kernel_initializer)(conv2)
    conv2 = BatchNormalization()(conv2) if use_bn else conv2
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer=kernel_initializer)(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer=kernel_initializer)(conv3)
    conv3 = BatchNormalization()(conv3) if use_bn else conv3
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer=kernel_initializer)(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer=kernel_initializer)(conv4)
    conv4 = BatchNormalization()(conv4) if use_bn else conv4
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer=kernel_initializer)(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer=kernel_initializer)(conv5)
    conv5 = BatchNormalization()(conv5) if use_bn else conv5
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same',
                 kernel_initializer=kernel_initializer)(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer=kernel_initializer)(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer=kernel_initializer)(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same',
                 kernel_initializer=kernel_initializer)(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer=kernel_initializer)(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer=kernel_initializer)(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same',
                 kernel_initializer=kernel_initializer)(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer=kernel_initializer)(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer=kernel_initializer)(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same',
                 kernel_initializer=kernel_initializer)(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer=kernel_initializer)(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer=kernel_initializer)(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same',
                   kernel_initializer=kernel_initializer)(conv9)
    conv10 = Conv2D(filters=1, kernel_size=1,
                    activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy',
                  metrics=[_mean_iou_unet_wrapper])

    model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


