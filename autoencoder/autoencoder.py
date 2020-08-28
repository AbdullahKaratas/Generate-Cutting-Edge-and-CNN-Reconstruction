from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Cropping2D
from tensorflow.keras.models import Model
import tensorflow as tf


class AE_Upsampling_Sample_TypeA(object):

    def __init__(self):
        # Encoding
        input_layer = Input(shape=(860, 564, 1))
        encoding_conv_layer_1 = Conv2D(
            32, (3, 3), activation='elu', padding='same'
        )(input_layer)
        encoding_pooling_layer_1 = MaxPooling2D(
            (2, 2), padding='same'
        )(encoding_conv_layer_1)
        encoding_conv_layer_2 = Conv2D(
            16, (3, 3), activation='elu', padding='same'
        )(encoding_pooling_layer_1)
        encoding_pooling_layer_2 = MaxPooling2D(
            (2, 2), padding='same'
        )(encoding_conv_layer_2)
        encoding_conv_layer_3 = Conv2D(
            16, (3, 3), activation='elu', padding='same'
        )(encoding_pooling_layer_2)
        code_layer = MaxPooling2D(
            (2, 2), padding='same'
        )(encoding_conv_layer_3)

        # Decoding
        decodging_conv_layer_1 = Conv2D(
            16, (3, 3), activation='elu', padding='same'
        )(code_layer)
        decodging_upsampling_layer_1 = UpSampling2D(
            (2, 2)
        )(decodging_conv_layer_1)
        decodging_conv_layer_2 = Conv2D(
            16, (3, 3), activation='elu', padding='same'
        )(decodging_upsampling_layer_1)
        decodging_upsampling_layer_2 = UpSampling2D(
            (2, 2)
        )(decodging_conv_layer_2)
        decodging_conv_layer_3 = Conv2D(
            32, (3, 3), activation='elu', padding='same'
        )(decodging_upsampling_layer_2)
        decodging_upsampling_layer_3 = UpSampling2D(
            (2, 2)
        )(decodging_conv_layer_3)
        output_layer = Conv2D(
            1, (3, 3), activation='sigmoid', padding='same'
        )(decodging_upsampling_layer_3)

        self._model = Model(input_layer, output_layer)
        self._model.compile(optimizer='adam', loss='MSE')

    def train(self, input_train, input_test, batch_size, epochs):
        self._model.fit(input_train,
                        input_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(
                            input_test,
                            input_test))

    def trainTFrecords(self, input_train, input_test, batch_size, epochs):
        self._model.fit(input_train,
                        epochs=epochs,
                        shuffle=True,
                        steps_per_epoch=10,
                        validation_data=input_test,
                        validation_steps=2)

    def trainTFrecords2(self, input_train, epochs):
        self._model.fit(input_train,
                        epochs=epochs,
                        shuffle=True,
                        steps_per_epoch=2)

    def getDecodedImage(self, encoded_imgs):
        decoded_image = self._model.predict(encoded_imgs)
        return decoded_image


class AE_Upsampling_Sample_TypeB(object):

    def __init__(self):
        # Encoding
        input_layer = Input(shape=(956, 948, 1))
        encoding_conv_layer_1 = Conv2D(
            32, (3, 3), activation='elu', padding='same'
        )(input_layer)
        encoding_pooling_layer_1 = MaxPooling2D(
            (2, 2), padding='same'
        )(encoding_conv_layer_1)
        encoding_conv_layer_2 = Conv2D(
            16, (3, 3), activation='elu', padding='same'
        )(encoding_pooling_layer_1)
        encoding_pooling_layer_2 = MaxPooling2D(
            (2, 2), padding='same'
        )(encoding_conv_layer_2)
        encoding_conv_layer_3 = Conv2D(
            16, (3, 3), activation='elu', padding='same'
        )(encoding_pooling_layer_2)
        code_layer = MaxPooling2D(
            (2, 2), padding='same'
        )(encoding_conv_layer_3)

        # Decoding
        decodging_conv_layer_1 = Conv2D(
            16, (3, 3), activation='elu', padding='same'
        )(code_layer)
        decodging_upsampling_layer_1 = UpSampling2D(
            (2, 2)
        )(decodging_conv_layer_1)
        decodging_conv_layer_2 = Conv2D(
            16, (3, 3), activation='elu', padding='same'
        )(decodging_upsampling_layer_1)
        decodging_upsampling_layer_2 = UpSampling2D(
            (2, 2)
        )(decodging_conv_layer_2)
        decodging_conv_layer_3 = Conv2D(
            32, (3, 3), activation='elu'
        )(decodging_upsampling_layer_2)
        decodging_upsampling_layer_3 = UpSampling2D(
            (2, 2)
        )(decodging_conv_layer_3)
        output_layer = Conv2D(
            1, (3, 3), activation='sigmoid', padding='same'
        )(decodging_upsampling_layer_3)

        self._model = Model(input_layer, output_layer)
        self._model.compile(optimizer='adam', loss='MSE')

    def train(self, input_train, input_test, batch_size, epochs):
        self._model.fit(input_train,
                        input_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(
                            input_test,
                            input_test))

    def trainTFrecords(self, input_train, input_test, batch_size, epochs):
        self._model.fit(input_train,
                        epochs=epochs,
                        shuffle=True,
                        steps_per_epoch=10,
                        validation_data=input_test,
                        validation_steps=2)

    def trainTFrecords2(self, input_train, epochs):
        self._model.fit(input_train,
                        epochs=epochs,
                        shuffle=True,
                        steps_per_epoch=2)

    def getDecodedImage(self, encoded_imgs):
        decoded_image = self._model.predict(encoded_imgs)
        return decoded_image


class AE_Upsampling_rezise(object):

    def __init__(self):
        # Encoding
        input_layer = Input(shape=(476, 476, 1))
        encoding_conv_layer_1 = Conv2D(
            16, (3, 3), activation='elu', padding='same'
        )(input_layer)
        encoding_pooling_layer_1 = MaxPooling2D(
            (2, 2), padding='same'
        )(encoding_conv_layer_1)

        encoding_conv_layer_2 = Conv2D(
            16, (3, 3), activation='elu', padding='same'
        )(encoding_pooling_layer_1)
        encoding_pooling_layer_2 = MaxPooling2D(
            (2, 2), padding='same'
        )(encoding_conv_layer_2)

        encoding_conv_layer_3 = Conv2D(
            32, (3, 3), activation='elu', padding='same'
        )(encoding_pooling_layer_2)
        encoding_pooling_layer_3 = MaxPooling2D(
            (2, 2), padding='same'
        )(encoding_conv_layer_3)

        encoding_conv_layer_4 = Conv2D(
            32, (3, 3), activation='elu', padding='same'
        )(encoding_pooling_layer_3)
        encoding_pooling_layer_4 = MaxPooling2D(
            (2, 2), padding='same'
        )(encoding_conv_layer_4)

        encoding_conv_layer_5 = Conv2D(
            64, (3, 3), activation='elu', padding='same'
        )(encoding_pooling_layer_4)
        code_layer = MaxPooling2D(
            (2, 2), padding='same'
        )(encoding_conv_layer_5)

        # Decoding
        decodging_conv_layer_1 = Conv2D(
            64, (3, 3), activation='elu', padding='same'
        )(code_layer)
        decodging_upsampling_layer_1 = UpSampling2D(
            (2, 2)
        )(decodging_conv_layer_1)

        decodging_conv_layer_2 = Conv2D(
            32, (3, 3), activation='elu', padding='same'
        )(decodging_upsampling_layer_1)
        decodging_upsampling_layer_2 = UpSampling2D(
            (2, 2)
        )(decodging_conv_layer_2)

        decodging_conv_layer_3 = Conv2D(
            32, (3, 3), activation='elu', padding='same'
        )(decodging_upsampling_layer_2)
        decodging_upsampling_layer_3 = UpSampling2D(
            (2, 2)
        )(decodging_conv_layer_3)
        decodging_upsampling_layer_3_Crop = Cropping2D(
            cropping=((1, 0), (0, 1))
        )(decodging_upsampling_layer_3)

        decodging_conv_layer_4 = Conv2D(
            16, (3, 3), activation='elu', padding='same'
        )(decodging_upsampling_layer_3_Crop)
        decodging_upsampling_layer_4 = UpSampling2D(
            (2, 2)
        )(decodging_conv_layer_4)

        decodging_conv_layer_5 = Conv2D(
            16, (3, 3), activation='elu', padding='same'
        )(decodging_upsampling_layer_4)
        decodging_upsampling_layer_5 = UpSampling2D(
            (2, 2)
        )(decodging_conv_layer_5)

        output_layer = Conv2D(
            1, (3, 3), activation='sigmoid', padding='same'
        )(decodging_upsampling_layer_5)

        self._model = Model(input_layer, output_layer)
        self._model.compile(optimizer='adam', loss='MSE')

    def train(self, input_train, input_test, batch_size, epochs):
        self._model.fit(input_train,
                        input_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(
                            input_test,
                            input_test))

    def trainTFrecords(self, input_train, input_test, batch_size, epochs):
        self._model.fit(input_train,
                        epochs=epochs,
                        shuffle=True,
                        steps_per_epoch=10,
                        validation_data=input_test,
                        validation_steps=2)

    def trainTFrecords2(self, input_train, epochs):
        self._model.fit(input_train,
                        epochs=epochs,
                        shuffle=True,
                        steps_per_epoch=2)

    def getDecodedImage(self, encoded_imgs):
        decoded_image = self._model.predict(encoded_imgs)
        return decoded_image
