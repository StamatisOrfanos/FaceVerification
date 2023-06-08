import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Input, Flatten

class EmbeddingLayer():
    def __init__(self, **kwargs):
        super().__init__()
    
    def embedding_model():
        # Input Layer
        input = Input(shape=(100,100,3), name='input_image')

        # Hidden layer 1
        convolution1 = Conv2D(filters=64, kernel_size=(10,10), activation='relu')(input)  
        maxPooling1  = MaxPooling2D(pool_size=(2,2), padding='same')(convolution1)

        # Hidden Layer 2
        convolution2 = Conv2D(filters=128, kernel_size=(7,7), activation='relu')(maxPooling1)  
        maxPooling2  = MaxPooling2D(pool_size=(2,2), padding='same')(convolution2)

        # Hidden Layer 3
        convolution3 = Conv2D(filters=128, kernel_size=(4,4), activation='relu')(maxPooling2)  
        maxPooling3  = MaxPooling2D(pool_size=(2,2), padding='same')(convolution3)

        # Output Layer
        convolution4 = Conv2D(filters=256, kernel_size=(4,4), activation='relu')(maxPooling3)  
        flatten  = Flatten()(convolution4)
        dense = Dense(4096, activation='sigmoid')(flatten)

        return Model(inputs=[input], outputs=[dense], name='embedding')


