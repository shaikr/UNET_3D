"""
Diogo Amorim, 2018-07-10
V-Net implementation in Keras 2
https://arxiv.org/pdf/1606.04797.pdf
"""

import functools

import tensorflow as tf
from keras import backend as K
from keras.layers import *
from keras import activations
from keras.layers.advanced_activations import PReLU
from keras.models import Model
from keras.optimizers import Adam
from ...metrics import dice_coefficient_loss, get_label_dice_coefficient_function, dice_coefficient, vod_coefficient

K.set_image_data_format("channels_first")

class Deconvolution3D(Layer):
    def __init__(self, filters, kernel_size, output_shape, subsample):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = (1,) + subsample + (1,)
        self.output_shape_ = output_shape
        assert K.backend() == 'tensorflow'
        super(Deconvolution3D, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 5
        self.input_shape_ = input_shape
        W_shape = self.kernel_size + (self.filters, input_shape[4],)
        self.W = self.add_weight(shape=W_shape,
                                 initializer=functools.partial(tf.keras.initializers.glorot_uniform()),
                                 name='{}_W'.format(self.name))
        self.b = self.add_weight(shape=(1, 1, 1, self.filters,), initializer='zero', name='{}_b'.format(self.name))
        self.built = True

    def compute_output_shape(self, input_shape):
        return (None,) + self.output_shape_[1:]

    def call(self, x, mask=None):
        return tf.nn.conv3d_transpose(x, self.W, output_shape=self.output_shape_,
                                      strides=self.strides, padding='SAME', name=self.name) + self.b

    def get_config(self):
        base_config = super(Deconvolution3D, self).get_config().copy()
        base_config['output_shape'] = self.output_shape_
        return base_config


def downward_layer(input_layer, n_convolutions, n_output_channels):
    inl = input_layer

    for _ in range(n_convolutions):
        inl = ReLU()(
            Conv3D(filters=(n_output_channels // 2), kernel_size=5,
                   padding='same', kernel_initializer='he_normal')(inl)
        )
    add_l = add([inl, input_layer])
    downsample = Conv3D(filters=n_output_channels, kernel_size=2, strides=2,
                        padding='same', kernel_initializer='he_normal')(add_l)
    downsample = ReLU()(downsample)
    return downsample, add_l


def upward_layer(input0, input1, n_convolutions, n_output_channels):
    merged = concatenate([input0, input1], axis=1)
    inl = merged
    for _ in range(n_convolutions):
        inl = ReLU()(
            Conv3D((n_output_channels * 4), kernel_size=5,
                   padding='same', kernel_initializer='he_normal')(inl)
        )
    add_l = add([inl, merged])
    shape = add_l.get_shape().as_list()
    new_shape = (1, shape[1] * 2, shape[2] * 2, shape[3] * 2, n_output_channels)
    #upsample = Deconvolution3D(n_output_channels, (2, 2, 2), new_shape, subsample=(2, 2, 2))(add_l)
    upsample = Conv3DTranspose(n_output_channels, kernel_size=(2, 2, 2), strides=(2, 2, 2), dilation_rate=2)(add_l)
    return ReLU()(upsample)


def vnet(input_shape, initial_learning_rate=0.0001, truth_index=0, truth_size=None,
         loss_function=dice_coefficient_loss, **kargs):

    metrics = ['binary_accuracy', vod_coefficient]
    if loss_function != dice_coefficient_loss:
        metrics += [dice_coefficient]
    # Layer 1
    inputs = Input(input_shape)
    conv1 = Conv3D(16, kernel_size=5, strides=1, padding='same', kernel_initializer='he_normal')(inputs)
    # conv1 = ReLU()(conv1)
    conv1 = ReLU()(conv1)
    print(conv1.get_shape())
    repeat1 = concatenate(16 * [inputs], axis=1)
    print(repeat1.get_shape())
    add1 = add([conv1, repeat1])
    print(add1.get_shape())
    down1 = Conv3D(32, 2, strides=2, padding='same', kernel_initializer='he_normal')(add1)
    down1 = ReLU()(down1)
    print(down1.get_shape())

    # Layer 2,3,4
    down2, add2 = downward_layer(down1, 2, 64)
    print(down2.get_shape())
    print(add2.get_shape())
    down3, add3 = downward_layer(down2, 3, 128)
    print(down3.get_shape())
    print(add3.get_shape())
    down4, add4 = downward_layer(down3, 3, 256)
    print(down4.get_shape())
    print(add4.get_shape())
    
    upsample_4 = Conv3DTranspose(64, kernel_size=(2, 2, 2), strides=(2, 2, 2), dilation_rate=2)(add4)
    # Layer 5,6,7
    upsample_5 = upward_layer(upsample_4, add3, 3, 32)
    upsample_6 = upward_layer(upsample_5, add2, 2, 16)
    print(f'ups_6: {upsample_6.get_shape()}')
    
    merged_9 = concatenate([upsample_6, add1], axis=1)
    conv_9_1 = Conv3D(32, kernel_size=(5, 5, 5), padding='same', kernel_initializer='he_normal')(merged_9)
    conv_9_1 = ReLU()(conv_9_1)
    add_9 = add([conv_9_1, merged_9])
    conv_9_2 = Conv3D(1, kernel_size=(1, 1, 1), padding='same', kernel_initializer='he_normal')(add_9)
    conv_9_2 = ReLU()(conv_9_2)
    
    # Layer 5
    # !Mudar kernel_size=(5, 5, 5) quando imagem > 64!
    #conv_5_1 = Conv3D(256, kernel_size=(5, 5, 5), padding='same', kernel_initializer='he_normal')(down4)
    #conv_5_1 = ReLU()(conv_5_1)
    #conv_5_2 = Conv3D(256, kernel_size=(5, 5, 5), padding='same', kernel_initializer='he_normal')(conv_5_1)
    #conv_5_2 = ReLU()(conv_5_2)
    #conv_5_3 = Conv3D(256, kernel_size=(5, 5, 5), padding='same', kernel_initializer='he_normal')(conv_5_2)
    #conv_5_3 = ReLU()(conv_5_3)
    #add5 = add([conv_5_3, down4])
    #aux_shape = add5.get_shape()
    #upsample_5 = Deconvolution3D(128, (2, 2, 2), output_shape=(None, aux_shape[1].value*2,aux_shape[2].value*2,
    #                                              aux_shape[3].value*2, 128), subsample=(2, 2, 2))(add5)
    #upsample_5 = Conv3DTranspose(128, kernel_size=(2, 2, 2), strides=(2, 2, 2), dilation_rate=2)(add5)                          
    #upsample_5 = ReLU()(upsample_5)
    #print(upsample_5.get_shape())

    # Layer 6,7,8
    #upsample_6 = upward_layer(upsample_5, add4, 3, 64)
    #upsample_7 = upward_layer(upsample_6, add3, 3, 32)
    #upsample_8 = upward_layer(upsample_7, add2, 2, 16)
    #print(f'ups_8: {upsample_8.get_shape()}')

    # Layer 9
    #merged_9 = concatenate([upsample_8, add1], axis=1)
    #conv_9_1 = Conv3D(32, kernel_size=(5, 5, 5), padding='same', kernel_initializer='he_normal')(merged_9)
    #conv_9_1 = ReLU()(conv_9_1)
    #add_9 = add([conv_9_1, merged_9])
    # conv_9_2 = Conv3D(1, kernel_size=(1, 1, 1), padding='same', kernel_initializer='he_normal')(add_9)
    #conv_9_2 = ReLU()(conv_9_2)

    # softmax = Softmax()(conv_9_2)
    sigmoid = Conv3D(1, kernel_size=(1, 1, 1), padding='same', kernel_initializer='he_normal',
                     activation='sigmoid')(conv_9_2)
    #sliced = Crop(-1, truth_index, truth_size)(sigmoid)

    model = Model(inputs=inputs, outputs=sigmoid)
    # model = Model(inputs=inputs, outputs=softmax)

    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=loss_function, metrics=metrics)
    return model
    

def Crop(dim, start, size, **kwargs):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]

    def func(x):
        dimension = dim
        if dimension == -1:
            dimension = len(x.shape) - 1
        if dimension == 0:
            return x[start:start+size]
        if dimension == 1:
            return x[:, start:start+size]
        if dimension == 2:
            return x[:, :, start:start+size]
        if dimension == 3:
            return x[:, :, :, start:start+size]
        if dimension == 4:
            return x[:, :, :, :, start:start+size]

    return Lambda(func, **kwargs)

