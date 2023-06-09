import tensorflow as tf
from tensorflow.keras.layers import Layer


class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()
    
    # Similarity calculation
    def sim_cal(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding-validation_embedding)




