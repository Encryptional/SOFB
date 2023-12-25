import keras
from keras.backend import sigmoid
from keras.layers import (LSTM, GRU, Activation, AveragePooling1D, Convolution1D, Dense, Dropout,MaxPooling1D,Conv1D,concatenate,
                          Flatten, Input, add, Dropout, Concatenate,GlobalMaxPool1D)
from keras.layers import BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, BatchNormalization, add, ZeroPadding1D, \
    Input, AveragePooling1D,UpSampling1D


def Conv_Block(inpt, nb_filter, kernel_size, strides=2, with_conv_shortcut=False):
    x = Conv1d_layer(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv1d_layer(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = Conv1d_layer(inpt, nb_filter=nb_filter, strides=3, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x

def Conv1d_layer(x, nb_filter, kernel_size, strides=2, padding='same'):
    x = Conv1D(nb_filter, kernel_size, padding=padding, strides=strides, activation='tanh')(x)
    x = BatchNormalization(axis=1)(x)
    return x

def ensemble_DLsequence_net():
    input3 = Input(shape=(1, 1024), name='dynamic')
    y = ZeroPadding1D(1)(input3)
    y = Conv1d_layer(y, nb_filter=32, kernel_size=3, strides=1, padding='valid')
    y = Dropout(0.4)(y)
    w = concatenate([y], axis=1)
    y = Conv_Block(y, nb_filter=32, kernel_size=3)
    y = Conv_Block(y, nb_filter=32, kernel_size=3)
    y = Conv_Block(y, nb_filter=32, kernel_size=3)
    y = Dropout(0.4)(y)
    w2 = concatenate([y, w], axis=1)
    y = Conv_Block(y, nb_filter=32, kernel_size=3, strides=2, with_conv_shortcut=True)
    y = Conv_Block(y, nb_filter=32, kernel_size=3)
    y = Conv_Block(y, nb_filter=32, kernel_size=3)
    y = Conv_Block(y, nb_filter=32, kernel_size=3)
    y = Dropout(0.4)(y)
    w3 = concatenate([y, w2], axis=1)
    y = Conv_Block(y, nb_filter=32, kernel_size=3, strides=3, with_conv_shortcut=True)
    y = Conv_Block(y, nb_filter=32, kernel_size=3)
    y = Conv_Block(y, nb_filter=32, kernel_size=3)
    y = Conv_Block(y, nb_filter=32, kernel_size=3)
    y = Conv_Block(y, nb_filter=32, kernel_size=3)
    y = Conv_Block(y, nb_filter=32, kernel_size=3)
    y = Dropout(0.4)(y)
    w4 = concatenate([y, w3], axis=1)
    y = Conv_Block(y, nb_filter=32, kernel_size=3, strides=4, with_conv_shortcut=True)
    y = Conv_Block(y, nb_filter=32, kernel_size=3)
    y = Conv_Block(y, nb_filter=32, kernel_size=3)
    y = Dropout(0.4)(y)
    time = concatenate([y, w4], axis=1)

    input2 = Input(shape=(1, 75), dtype='float', name='bio')
    concat = concatenate([y, input2], axis=-1)
    z = BatchNormalization(axis=-1)(concat)
    kernel_list = [1, 3, 5, 7] 
    p = []
    for i in kernel_list:
        c_l = Conv1D(filters=64, kernel_size=i, activation='relu', padding='same')(z)
        p.append(c_l)
    concat2 = concatenate([i for i in p], axis=-1)
    GL = BatchNormalization(axis=-1)(concat2)

    lstm = Bidirectional(LSTM(64, return_sequences=True))(time)
    fr = LSTM(64, return_sequences=False)(lstm)

    input1 = Input(shape=(1, 1024), dtype='float', name='generalization')
    norm = BatchNormalization(axis=-1)(input1)
    fs = concatenate([norm, GL], axis=-1)
    x0 = BatchNormalization(axis=-1)(fs)
    x1 = Bidirectional(LSTM(64, return_sequences=True))(x0)
    drop1 = Dropout(0.5)(x1)
    flat = Flatten()(drop1)
    
    fi = concatenate([flat, fr], axis=-1)
    outputs = Dense(units=2, activation='softmax', name='outputs')(fi)

    model = Model(inputs=[input1, input2, input3], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

    return model

