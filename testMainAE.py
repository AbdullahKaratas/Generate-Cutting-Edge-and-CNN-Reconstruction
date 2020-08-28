import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
#from autoencoder.autoencoder import AutoencoderUpsampling as AEUpsampling
from tensorflow.keras.models import load_model
from PIL import Image


def main():

    model = load_model('resized_model_both_smaller_BN.h5')
    print(model.summary())
    Z_test = np.load('./data/sample_TypeB/Test_resize.npy')
    Z_test = np.reshape(
        Z_test, (Z_test.shape[0], Z_test.shape[1], Z_test.shape[2], 1))

    Z_predict = model.predict(Z_test)
    Z_predict = np.reshape(
        Z_predict, (Z_predict.shape[0], Z_predict.shape[1], Z_predict.shape[2], 1))

    x = np.linspace(0, 1, Z_test.shape[1])
    y = np.linspace(0, 1, Z_test.shape[2])
    X, Y = np.meshgrid(x, y)
    # resize also X and Y
    Z_predict_surf = np.resize(Z_predict[0, :, :], new_shape=(476, 476))
    Z_test_surf = np.resize(Z_test[0, :, :], new_shape=(476, 476))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(
        X, Y, Z_test_surf, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    surf2 = ax.plot_surface(
        X, Y, Z_predict_surf, cmap=cm.gray, linewidth=0, antialiased=False)
    plt.show()


main()
