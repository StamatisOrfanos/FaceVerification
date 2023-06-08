from L1Dist import L1Dist
from embedding_layer import EmbeddingLayer
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

class Siamese():
    def __init__(self, **kwargs):
        super().__init__()
    
    def siamese_model():

        # Input Layer (input_image = Anchor,  validation_image=(Positive | Negative))
        input_image = Input(name='input_image', shape=(100,100,3))
        validation_image = Input(name='validation_image', shape=(100,100,3))

        # Create the Embedding + L1 Distance between the two input images
        embedding = EmbeddingLayer.embedding_model()
        l1Dist_model = L1Dist()
        distances = l1Dist_model(embedding(input_image), embedding(validation_image))

        # Classification Layer
        classifier = Dense(1, activation="sigmoid")(distances)

        return Model(inputs=[input_image, validation_image], outputs=[classifier], name='SiameseModel')