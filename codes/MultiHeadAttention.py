#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/1/28 9:30
@File:          MultiHeadAttention.py
'''

import math
import tensorflow as tf
from keras import backend as K
from keras.layers import Layer, Dense
from keras import constraints, initializers, regularizers

class MultiHeadAttention(Layer):
    def __init__(self,
                 num_heads,
                 key_dim,
                 value_dim=None,
                 out_dim=None,
                 dropout=0.0,
                 return_attention_scores=False,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim if value_dim is not None else key_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.return_attention_scores =return_attention_scores
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        super(MultiHeadAttention, self).build(input_shape)
        common_kwargs = dict(
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint)
        out_dim = self.out_dim if self.out_dim is not None else input_shape[0][-1]
        self.query_fc = Dense(self.num_heads * self.key_dim, **common_kwargs)
        self.key_fc = Dense(self.num_heads * self.key_dim, **common_kwargs)
        self.value_fc = Dense(self.num_heads * self.value_dim, **common_kwargs)
        self.out_fc = Dense(out_dim, **common_kwargs)

    def call(self, inputs, mask=None, training=None, **kwargs):
        self._validate_call_args(inputs, mask)
        q, v = inputs[:2]
        k = inputs[2] if len(inputs) > 2 else v
        if mask is not None:
            q_mask, v_mask = mask
        else:
            q_mask = v_mask = None

        q = self.query_fc(q)
        k = self.key_fc(k)
        v = self.value_fc(v)

        q = K.reshape(q, (-1, K.shape(q)[1], self.num_heads, self.key_dim))
        k = K.reshape(k, (-1, K.shape(k)[1], self.num_heads, self.key_dim))
        v = K.reshape(v, (-1, K.shape(v)[1], self.num_heads, self.value_dim))

        result, attention_scores = self._compute_attention(q, k, v, attention_mask=v_mask, training=training)

        result = K.reshape(result, (-1, K.shape(result)[1], self.num_heads * self.value_dim))
        result = self.out_fc(result)
        if q_mask is not None:
            q_mask = K.expand_dims(q_mask)
            result *= K.cast(q_mask, K.dtype(result))
        if self.return_attention_scores:
            return [result, attention_scores]
        return result

    def compute_output_shape(self, input_shape):
        b, Tq, dim = input_shape[0]
        out_dim = self.out_dim if self.out_dim is not None else dim
        Tv = input_shape[1][1]
        if self.return_attention_scores:
           return [(b, Tq, out_dim), (b, self.num_heads, Tq, Tv)]

        return (b, Tq, out_dim)

    def compute_mask(self, inputs, mask=None):
        self._validate_call_args(inputs=inputs, mask=mask)
        if mask is not None:
            q_mask = mask[0]
            if q_mask is None:
                return None
            return q_mask
        return None

    def get_config(self):
        config = {
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'value_dim': self.value_dim,
            'dropout': self.dropout,
            'return_attention_scores': self.return_attention_scores,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _validate_call_args(self, inputs, mask):
        class_name = self.__class__.__name__
        if not isinstance(inputs, list):
            raise ValueError(
                '{} layer must be called on a list of inputs, namely [query, value] '
                'or [query, value, key].'.format(class_name))
        if len(inputs) < 2 or len(inputs) > 3:
            raise ValueError(
                '{} layer accepts inputs list of length 2 or 3, '
                'namely [query, value] or [query, value, key]. '
                'Given length: {}'.format(class_name, len(inputs)))
        if mask is not None:
            if not isinstance(mask, list):
                raise ValueError(
                    '{} layer mask must be a list, '
                    'namely [query_mask, value_mask].'.format(class_name))
            if len(mask) < 2 or len(mask) > len(inputs):
                raise ValueError(
                    '{} layer mask must be a list of length 2, namely [query_mask, '
                    'value_mask]. Given length: {}'.format(class_name, len(mask)))

    def _compute_attention(self, query, key, value, attention_mask=None, training=None):
        query = query / math.sqrt(self.key_dim)
        attention_scores = tf.einsum('bqnh, bvnh->bnqv', query, key)
        weights = self._masked_softmax(attention_scores, attention_mask)
        if training is None:
            training = K.learning_phase()
        def dropped_weights():
            return K.dropout(weights, self.dropout)
        weights = K.in_train_phase(dropped_weights, weights, training=training)
        return tf.einsum('bnqv, bvnh->bqnh', weights, value), weights

    def _masked_softmax(self, attention_scores, attention_mask):
        if attention_mask is not None:
            expand_dims_times = K.ndim(attention_scores) - K.ndim(attention_mask)
            for _ in range(expand_dims_times):
                attention_mask = K.expand_dims(attention_mask, axis=1)
            padding_mask = tf.logical_not(attention_mask)
            attention_scores -= 1e12 * K.cast(padding_mask, K.dtype(attention_scores))
        return K.softmax(attention_scores)