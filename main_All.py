# Main Libarys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

# Help Libarys
import os
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import time
from sklearn.utils import shuffle

# Autoencoders
from autoencoder.autoencoder import AE_Upsampling_Sample_TypeA as AE_Upsampling_Sample_TypeA
from autoencoder.autoencoder import AE_Upsampling_Sample_TypeB as AE_Upsampling_Sample_TypeB
from autoencoder.autoencoder import AE_Upsampling_rezise as AE_Upsampling_rezise

# Own Help Libarys
from main import main as generatingData
from mainAE import main as callAutoencoder
#from testMainAE import  main as testResult

# Settings
''' 
- sample = 'both', sample = 'TypeA', sample ='TypeB' 
              ||  If 'both', The images are scaled and partly cropped. resizing will always be True
- nTimes = [Number of Trainings data, Number of Validation data, Number of Test data]
- name = [Name of Trainings data file, Name of Validation data file, Name of Test data file]
- SavePath = Saving Path, only one Path is possibles
- resizing = False as default
- threading = False, threading = True
'''


run_sample = 'both'
# run_sample = 'TypeA'
# run_sample = 'TypeB'

savePredictionName = 'Prediction_for_Test_Data_resized_both'
saveModelName = 'resized_model_both_smaller_BN'

name = ['Train_rezise', 'Val_resize', 'Test_resize']
SavePath_sample_TypeA = './data/sample_TypeA/'
SavePath_sample_TypeB = './data/sample_TypeB/'

nTimes = [100, 20, 3]

resizing = True

if run_sample == 'both':
    autoencoderModel = 'AE_Upsampling_rezise'
    resizing = True
    generatingData(sample='04', nTimes=nTimes, name=name,
                   SavePath=SavePath_sample_TypeA, resizing=resizing, threading=False)
    generatingData(sample='11', nTimes=nTimes, name=name,
                   SavePath=SavePath_sample_TypeB, resizing=resizing, threading=False)
    training_data_04 = np.load(SavePath_sample_TypeA + name[0] + '.npy')
    training_data_11 = np.load(SavePath_sample_TypeB + name[0] + '.npy')

    training_data = np.vstack(
        (training_data_04, training_data_11)
    )
    training_data = shuffle(training_data, random_state=0)

    val_data_04 = np.load(SavePath_sample_TypeA + name[1] + '.npy')
    val_data_11 = np.load(SavePath_sample_TypeB + name[1] + '.npy')
    val_data = np.vstack(
        (val_data_04, val_data_11)
    )
    val_data = shuffle(val_data, random_state=0)

    test_data_04 = np.load(SavePath_sample_TypeA + name[2] + '.npy')
    test_data_11 = np.load(SavePath_sample_TypeB + name[2] + '.npy')
    test_data = np.vstack(
        (test_data_04, test_data_11)
    )
    test_data = shuffle(test_data, random_state=0)

    files = np.vstack(
        (training_data, val_data, test_data)
    )
    filesSize = [training_data.shape[0], val_data.shape[0], test_data.shape[0]]

    callAutoencoder(files, autoencoderModel=autoencoderModel, savePredictionName=savePredictionName,
                    saveModelName=saveModelName, num_epochs=50, batch_size=5, filesSize=filesSize)

elif run_sample == '04':
    if resizing == False:
        autoencoderModel = 'AE_Upsampling_Sample_TypeA'
    else:
        autoencoderModel = 'AE_Upsampling_rezise'

    generatingData(sample='04', nTimes=nTimes, name=name,
                   SavePath=SavePath_sample_TypeA, resizing=resizing, threading=False)
    training_data = np.load(SavePath_sample_TypeA + name[0] + '.npy')
    val_data = np.load(SavePath_sample_TypeA + name[1] + '.npy')
    test_data = np.load(SavePath_sample_TypeA + name[2] + '.npy')

    training_data = training_data[:, :860, :564]
    val_data = val_data[:, :860, :564]
    test_data = test_data[:, :860, :564]

    files = np.vstack(
        (training_data, val_data, test_data)
    )
    filesSize = [training_data.shape[0], val_data.shape[0], test_data.shape[0]]
    callAutoencoder(files, autoencoderModel=autoencoderModel, savePredictionName=savePredictionName,
                    saveModelName=saveModelName, num_epochs=50, batch_size=5, filesSize=filesSize)

elif run_sample == '11':
    if resizing == False:
        autoencoderModel = 'AE_Upsampling_Sample_TypeB'
    else:
        autoencoderModel = 'AE_Upsampling_rezise'

    generatingData(sample='11', nTimes=nTimes, name=name,
                   SavePath=SavePath_sample_TypeB, resizing=resizing, threading=False)
    training_data = np.load(SavePath_sample_TypeB + name[0] + '.npy')
    val_data = np.load(SavePath_sample_TypeB + name[1] + '.npy')
    test_data = np.load(SavePath_sample_TypeB + name[2] + '.npy')

    training_data = training_data[:, :956, :948]
    val_data = val_data[:, :956, :948]
    test_data = test_data[:, :956, :948]

    files = np.vstack(
        (training_data, val_data, test_data)
    )
    filesSize = [training_data.shape[0], val_data.shape[0], test_data.shape[0]]
    callAutoencoder(files, autoencoderModel=autoencoderModel, savePredictionName=savePredictionName,
                    saveModelName=saveModelName, num_epochs=50, batch_size=5, filesSize=filesSize)

# Das wir an der Memoline erkennen, dass es eine Inv Nummer ist.
# Machien Learning ansetzen
# => Eine art nachbearbeitung.
# => Invoice
