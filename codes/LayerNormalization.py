#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/1/29 18:59
@File:          LayerNormalization.py
'''

from keras import backend as K
from keras.layers import Layer

class LayerNormalization(Layer):
    def __init__(self, center=True, scale=True, epsilon=None, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.center = center
        self.scale = scale
        self.epsilon = epsilon or 1e-12

    def build(self, input_shape):
        super(LayerNormalization, self).build(input_shape)
        shape = (input_shape[-1],)

        if self.center:
            self.beta = self.add_weight(shape=shape, initializer='zeros', name='beta')
        if self.scale:
            self.gamma = self.add_weight(shape=shape, initializer='ones', name='gamma')


    def call(self, inputs, **kwargs):
        if self.center:
            beta = self.beta
        if self.scale:
            gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = K.mean(outputs, axis=-1, keepdims=True)
            outputs = outputs - mean
        if self.scale:
            variance = K.mean(K.square(outputs), axis=-1, keepdims=True)
            std = K.sqrt(variance + self.epsilon)
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))