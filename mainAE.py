import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from autoencoder.autoencoder import AE_Upsampling_Sample_TypeA as AE_Upsampling_Sample_TypeA
from autoencoder.autoencoder import AE_Upsampling_Sample_TypeB as AE_Upsampling_Sample_TypeB
from autoencoder.autoencoder import AE_Upsampling_rezise as AE_Upsampling_rezise
import cv2
import time


def main(filenames, autoencoderModel, savePredictionName, saveModelName, num_epochs=50, batch_size=5, filesSize=[]):
    # Check
    tf.random.set_seed(22)
    np.random.seed(22)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    assert tf.__version__.startswith('2.')

    # Load Numpy Data -> current solution
    if isinstance(filenames[0], str):
        Z_train = np.load(filenames[0])
        Z_val = np.load(filenames[1])
        Z_test = np.load(filenames[2])
        print('All data are loaded')
    else:  # To do, check is filesSize is not empty
        Z_train = filenames[:filesSize[0], :, :]
        Z_val = filenames[filesSize[0]:(filesSize[0]+filesSize[1]), :, :]
        Z_test = filenames[(filesSize[0]+filesSize[1]):, :, :]

    Z_train_crop, Z_val_crop,  Z_test_crop = Z_train.astype(
        np.float32), Z_val.astype(np.float32), Z_test.astype(np.float32)

    Z_train_crop = np.reshape(Z_train_crop, (len(
        Z_train_crop), Z_train_crop.shape[1], Z_train_crop.shape[2], 1))
    Z_val_crop = np.reshape(
        Z_val_crop, (len(Z_val_crop), Z_val_crop.shape[1], Z_val_crop.shape[2], 1))
    Z_test_crop = np.reshape(
        Z_test_crop, (len(Z_test_crop), Z_test_crop.shape[1], Z_test_crop.shape[2], 1))

    if autoencoderModel == 'AE_Upsampling_rezise':
        autoencoder = AE_Upsampling_rezise()
        print(autoencoder._model.summary())
    elif autoencoderModel == 'AE_Upsampling_Sample_TypeB':
        autoencoder = AE_Upsampling_Sample_TypeB()
        print(autoencoder._model.summary())
    elif autoencoderModel == 'AE_Upsampling_Sample_TypeA':
        autoencoder = AE_Upsampling_Sample_TypeA()
        print(autoencoder._model.summary())

    # Convolutional implementation
    autoencoder.train(Z_train_crop, Z_val_crop, batch_size, num_epochs)
    decoded_imgs = autoencoder.getDecodedImage(Z_test_crop)
    # print(decoded_imgs)
    np.save(savePredictionName + '.npy', decoded_imgs)
    autoencoder._model.save(saveModelName + '.h5')
