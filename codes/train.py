#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/1/28 15:45
@File:          train.py
'''

from keras import backend as K
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers import *
from keras.optimizers import Adam
from keras import Model

from PositionEmbedding import SinusoidalPositionEmbedding
from MultiHeadAttention import MultiHeadAttention
from LayerNormalization import LayerNormalization

max_words = 20000
maxlen = 100
embed_dim = 64
batch_size = 128

(x_train, y_train), (x_val, y_val) = imdb.load_data(num_words=max_words)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = sequence.pad_sequences(x_val, maxlen=maxlen)

text_input = Input(shape=(maxlen, ), dtype='int32')
x = Embedding(max_words, embed_dim)(text_input)
x = SinusoidalPositionEmbedding()(x)

def transformer_encoder(inputs, num_heads=4, dropout_rate=0.1):
    in_dim = K.int_shape(inputs)[-1]
    x = MultiHeadAttention(num_heads, in_dim)([inputs, inputs])
    x = Dropout(dropout_rate)(x)
    x = add([inputs, x])
    x1 = LayerNormalization()(x)
    x = Dense(in_dim * 2, activation='relu')(x1)
    x = Dense(in_dim)(x)
    x = Dropout(dropout_rate)(x)
    x = add([x1, x])
    x = LayerNormalization()(x)
    return x

x = transformer_encoder(x)
x = GlobalAveragePooling1D()(x)
out = Dense(1, activation='sigmoid')(x)

model = Model(text_input, out)
model.compile(Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=10, validation_data=(x_val, y_val))