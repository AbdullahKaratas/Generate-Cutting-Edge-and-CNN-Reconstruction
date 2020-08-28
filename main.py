import numpy as np
import tools as tl
import tensorflow as tf
from scipy.interpolate import UnivariateSpline
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

from threading import Thread
import time


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    # Change value = [value] in value = value.reshape(-1)
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(image_float):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    feature = {
        'imageValue': _float_feature(image_float),
    }

    # Create a Features message using tf.train.Example.
    return tf.train.Example(features=tf.train.Features(feature=feature))


def _dtype_feature(ndarray):
    """match appropriate tf.train.Feature class with dtype of ndarray"""
    assert isinstance(ndarray, np.ndarray)
    dtype_ = ndarray.dtype
    if dtype_ == np.float64 or dtype_ == np.float32:
        return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
    elif dtype_ == np.int64 or dtype_ == np.int32:
        return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))
    else:
        raise ValueError(
            "The input should be numpy ndarray. Instead got {}".format(ndarray.dtype))


def generatingData(sample, nTimes=20000, name='No_Name_was_given', SavePath='./', resizing=False, defects=False):

    # Path of File.
    pathOfFile = './masterdata/'
    if sample == 'TypeA':
        FILES = ['sample_typeA.txt']
    elif sample == 'TypeB':
        FILES = ['sample_typeB.txt']

    # Initial Load
    filename = pathOfFile + FILES[0]
    Z_initial = np.loadtxt(filename, delimiter=',')  # Load Data for Z
    numberofrows, numberofcols = np.shape(Z_initial)
    # ALL x Values but not all y values
    Z_initial_NEW = Z_initial[30:(numberofrows - 30), :]

    # start_time = time.time()
    if resizing == False:
        ZSAVE = np.zeros(
            (nTimes, Z_initial_NEW.shape[0], Z_initial_NEW.shape[1]), dtype=np.float32)
    else:
        # x_shape_temp = int(Z_initial_NEW.shape[0] / 2)
        # y_shape_temp = int(Z_initial_NEW.shape[1] / 2)
        x_shape_temp = 476
        y_shape_temp = 476

        ZSAVE = np.zeros((nTimes, x_shape_temp, y_shape_temp),
                         dtype=np.float32)

    for ijk in range(0, nTimes):
        dx, dy = tl.dxdy(FILES[0])
        Z = np.loadtxt(filename, delimiter=',')  # Load Data for Z
        numberofrows, numberofcols = np.shape(Z)
        X = tl.generatex(dx, numberofrows, numberofcols)
        Y = tl.generatey(dy, numberofrows, numberofcols)

        # Reduce the data
        #   => There can be lot of data with NaN only.
        # ALL x Values but not all y values
        XNEW = X[30:(numberofrows - 30), :]
        # ALL x Values but not all y values
        YNEW = Y[30:(numberofrows - 30), :]
        # ALL x Values but not all y values
        ZNEW = Z[30:(numberofrows - 30), :]

        # Everything is still okay

        # Step 1: Find max Values and start a paramterisation
        maxofXYZ, amaxv = tl.maxpointsXYZ(XNEW, YNEW, ZNEW)
        t = tl.parametrisierung(maxofXYZ)
        opt = 0.000000032
        opt = np.random.uniform(opt-.000000005, opt + .000000005)
        sx, sy, sz = tl.smoothingdata(
            maxofXYZ, t, opt)  # sx verwenden für Maximum

        #ax.plot_wireframe(XNEW, YNEW, ZNEW)
        #ax.plot(sx, sy, sz, 'r')
        # plt.show()

        ySlice = 30  # Choose a arbitary cut.

        xTemp = XNEW[ySlice, :]
        zTemp = ZNEW[ySlice, :]
        ts = np.argwhere(np.isnan(zTemp) == 0)
        ts = ts.ravel()

        xTemp = xTemp[ts]
        zTemp = zTemp[ts]
        zTemp = tl.smooth(zTemp)
        originalMaxValueZTemp = np.max(zTemp)
        zTemp = zTemp / np.max(zTemp)
        xTempOrigin = np.copy(xTemp)

        iMax = np.argmin(abs(xTemp - sx[ySlice]))
        iMaxOrigin = np.copy(iMax)
        dzTemp = tl.derivativ(xTemp, zTemp)
        dzTemp = tl.smooth(dzTemp, window_len=20)

        ddzTemp = tl.derivativ(xTemp, dzTemp)
        ddzTemp = tl.smooth(ddzTemp, window_len=20)

        # Maximum of 6 Points.
        point = tl.generatePoints(ddzTemp, dzTemp, zTemp, iMax, nrBins=5)
        pointsOrigin = np.copy(point)

        # Step 3 Estimate the polynoms
        zSim = tl.getZSim(point, iMax, xTemp, zTemp)
        zSim = np.copy(zSim * originalMaxValueZTemp)

        #plt.plot(xTemp, zTemp)
        #plt.plot(xTemp, dzTemp/np.nanmax(np.abs(dzTemp)))
        #plt.plot(xTemp, ddzTemp/np.nanmax(np.abs(ddzTemp)))
        #plt.plot(xTemp[point], zTemp[point], 'ro')
        # plt.show()

        argmaxOfSimulatedProfil = np.nanargmax(zSim)
        maxOfSimulatedProfil = np.nanmax(zSim)
        lengthOfSimulatedProfil = zSim.shape[0]

        XSIM = XNEW
        YSIM = YNEW
        #ZSIM = np.nan * np.zeros( ZNEW.shape )
        ZSIM = np.zeros(ZNEW.shape)

        # Dataset is reduce because of invlaid points in corner
        for i in range(0, XNEW.shape[0]):
            xTemp = XNEW[i, :]
            iMax = np.argmin(abs(xTemp - sx[i]))
            if xTemp.shape[0] >= iMax + lengthOfSimulatedProfil - argmaxOfSimulatedProfil:
                lastValue = iMax + lengthOfSimulatedProfil - argmaxOfSimulatedProfil
            else:
                lastValue = xTemp.shape[0]

            deltaInX = iMax - argmaxOfSimulatedProfil
            while deltaInX < 0:
                zSimTemp = tl.getZSim(
                    pointsOrigin, iMaxOrigin, xTempOrigin, zTemp)
                zSimTemp = np.copy(zSimTemp * originalMaxValueZTemp)
                argmaxOfSimulatedProfil = np.nanargmax(zSimTemp)
                deltaInX = iMax - argmaxOfSimulatedProfil
                print('in while loop caused of deltaInX = ', deltaInX)

            deltaInZ = sz[i] - maxOfSimulatedProfil
            ZSIM[i, deltaInX:lastValue] = zSim[0:lastValue-deltaInX] + deltaInZ

        ZSIM = tl.smoothZSim(YSIM, ZSIM, sv=0e-08)
        # ax.plot_surface(XNEW, YNEW, ZSIM, rstride=15, cstride=15, color='green')
        # plt.show()
        if defects == True:
            # Defects
            indexLength = 50
            depth = 0.001 * originalMaxValueZTemp
            Defect1 = tl.generateDefect(point, iMaxOrigin, xTemp, indexLength, depth,
                                        dy, XNEW, sx, argmaxOfSimulatedProfil, startDefect=0.05, shift='on')

            indexLength = 90
            depth = 0.0007 * originalMaxValueZTemp
            Defect3 = tl.generateDefect(point, iMaxOrigin, xTemp, indexLength, depth, dy, XNEW,
                                        sx, argmaxOfSimulatedProfil, startDefect=0.6, maxWidthDefect=0.4, minWidthDefect=0)

            indexLength = 90
            depth = 0.0007 * originalMaxValueZTemp
            Defect4 = tl.generateDefect(point, iMaxOrigin, xTemp, indexLength, depth, dy, XNEW,
                                        sx, argmaxOfSimulatedProfil, startDefect=0.7, maxWidthDefect=0.1, minWidthDefect=0.1)

            indexLength = 140
            depth = 0.0004 * originalMaxValueZTemp
            Defect5 = tl.generateDefect(point, iMaxOrigin, xTemp, indexLength, depth, dy, XNEW, sx,
                                        argmaxOfSimulatedProfil, startDefect=0.2, maxWidthDefect=0.2, minWidthDefect=0.1, shift='on')

            indexLength = 70
            depth = 0.0005 * originalMaxValueZTemp
            Defect6 = tl.generateDefect(point, iMaxOrigin, xTemp, indexLength, depth, dy, XNEW, sx,
                                        argmaxOfSimulatedProfil, startDefect=0.45, maxWidthDefect=0.13, minWidthDefect=0, shift='on')

            indexLength = 90
            depth = 0.0007 * originalMaxValueZTemp
            Defect7 = tl.generateDefect(point, iMaxOrigin, xTemp, indexLength, depth, dy, XNEW, sx,
                                        argmaxOfSimulatedProfil, startDefect=0.8, maxWidthDefect=0.3, minWidthDefect=0, shift='on')

            #ZSIMWithDefect = ZSIM + Defect1
            ZSIM = ZSIM + Defect1 + Defect3 + Defect4 + Defect5 + Defect6 + Defect7
        else:
            ZSIM = ZSIM

        #ZSIMWithDefect = ZSIM
        xLength = ZSIM.shape[0]
        yLength = ZSIM.shape[1]

        # Add Noice:
        for j in range(0, xLength):
            ZSIM[j, :] = ZSIM[j, :] + \
                np.random.uniform(low=-2e-06, high=+2e-06)

        for i in range(0, yLength):
            ZSIM[:, i] = ZSIM[:, i] + \
                np.random.uniform(low=-2e-06, high=+2e-06)

        # Between 0 and 1
        ZSIM = (ZSIM - np.min(np.min(ZSIM))) / \
            (np.max(np.max(ZSIM)) - np.min(np.min(ZSIM)))

        # Resize the ZSAVE now
        ZSIM = ZSIM.astype(np.float32)

        if resizing == False:
            ZSAVE[ijk, :, :] = ZSIM
        else:
            ZSIM_temp = np.copy(ZSIM)
            ZSIM_resize = np.array(Image.fromarray(
                ZSIM_temp).resize(size=(y_shape_temp, x_shape_temp)))
            if ZSIM_resize.shape[0] > 476:
                if ZSIM_resize.shape[1] > 476:
                    ZSIM_resize_crop = ZSIM_resize[0:476, 0:476]
                else:
                    ZSIM_resize_crop = np.zeros((476, 476))
                    ZSIM_resize_crop[:, 0:ZSIM_resize.shape[1]
                                     ] = ZSIM_resize[0:476, :]
            else:
                ZSIM_resize_crop = np.zeros((476, 476))
                if ZSIM_resize.shape[1] > 476:
                    ZSIM_resize_crop[0:ZSIM_resize.shape[0],
                                     :] = ZSIM_resize[: 0:476]
                else:
                    ZSIM_resize_crop[0:ZSIM_resize.shape[0],
                                     0:ZSIM_resize.shape[1]] = ZSIM_resize

            ZSAVE[ijk, :, :] = ZSIM_resize_crop

        #fig = plt.figure()
        #ax = fig.gca(projection='3d')
        #surface = ax.plot_surface(XSIM, YSIM, ZSIM, cmap='Greys', linewidth=0.25, vmin = -1.5, vmax = 1)
        # plt.show()

        #print(time.time() - start_time)
        print('---- Done --- ', ijk / nTimes * 100)

    # Resize the ZSAVE now
    print('Now saving as ZSAVE')
    save_npy_in = SavePath + name + '.npy'
    np.save(save_npy_in, ZSAVE)
    print('npy is saved')
    print('Now saving TFrecords ...')
    file_path = SavePath + name + '.tfrecords'

    X_flat = np.reshape(ZSAVE, [ZSAVE.shape[0], np.prod(ZSAVE.shape[1:])])
    dtype_X = _dtype_feature(X_flat)

    with tf.io.TFRecordWriter(file_path) as writer:

        for idx in range(X_flat.shape[0]):

            x = X_flat[idx]
            x_sh = np.asarray(ZSAVE.shape[1:])
            dtype_xsh = _dtype_feature(x_sh)

            d_feature = {}
            d_feature["X"] = dtype_X(x)
            d_feature["x_shape"] = dtype_xsh(x_sh)

            features = tf.train.Features(feature=d_feature)
            example = tf.train.Example(features=features)
            serialized = example.SerializeToString()
            writer.write(serialized)


def main(sample, nTimes, name, SavePath, resizing=False, threading=False):
    generatingData(
        sample, nTimes=nTimes[0], name=name[0], SavePath=SavePath, resizing=resizing)
    generatingData(
        sample, nTimes=nTimes[1], name=name[1], SavePath=SavePath, resizing=resizing)
    generatingData(sample, nTimes=nTimes[2], name=name[2],
                   SavePath=SavePath, resizing=resizing, defects=True)

    #t1 = Thread(target=generatingData, args=[7000, 'Train_Images', True])
    #t2 = Thread(target=generatingData, args=[1400, 'Val_Images', True])
    #t3 = Thread(target=generatingData, args=[10, 'Test_Images', True])

    # t1.start()
    # t2.start()
    # t3.start()
