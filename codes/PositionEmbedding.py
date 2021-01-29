#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/1/28 9:33
@File:          PositionEmbedding.py
'''

import tensorflow as tf
from keras import backend as K
from keras.layers import Layer
from keras.initializers import Zeros

class PositionEmbedding(Layer):
    def __init__(self, **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(PositionEmbedding, self).build(input_shape)
        self.embeddings = self.add_weight(
            name='embeddings',
            shape=input_shape[1:],
            initializer=Zeros()
        )

    def call(self, inputs, **kwargs):
        positions = K.arange(0, stop=K.int_shape(inputs)[1])[None]
        positions = K.gather(self.embeddings, positions)
        return inputs + positions

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return mask

class SinusoidalPositionEmbedding(Layer):
    def __init__(self, **kwargs):
        super(SinusoidalPositionEmbedding, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, **kwargs):
        seq_len, out_dim = K.int_shape(inputs)[1:]
        positions = K.arange(0, stop=seq_len, dtype=self.dtype)[None]
        indices = K.arange(0, stop=out_dim // 2, dtype=self.dtype)
        indices = K.pow(10000.0, -2 * indices / out_dim)
        positions = tf.einsum('bn,d->bnd', positions, indices)
        positions = K.stack([K.sin(positions), K.cos(positions)], axis=-1)
        positions = K.reshape(positions, (-1, seq_len, out_dim))

        return inputs + positions

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return mask