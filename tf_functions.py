import datetime
import copy
import numpy as np
from collections import namedtuple
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras import Model, layers, Input

class ImagePool(object):

    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            index = int(np.random.rand()*self.maxsize)
            tmp1 = copy.copy(self.images[index])[0]
            self.images[index][0] = image[0]
            index = int(np.random.rand()*self.maxsize)
            tmp2 = copy.copy(self.images[index])[1]
            self.images[index][1] = image[1]
            return [tmp1, tmp2]
        else:
            return image

def load_npy_data(npy_data):
    npy_A = np.load(npy_data[0]) * 1.  # 64 * 84 * 1
    npy_B = np.load(npy_data[1]) * 1.  # 64 * 84 * 1
    npy_AB = np.concatenate((npy_A.reshape(npy_A.shape[0], npy_A.shape[1], 1),
                             npy_B.reshape(npy_B.shape[0], npy_B.shape[1], 1)),
                            axis=2)  # 64 * 84 * 2
    return npy_AB

def get_now_datetime():
    now = datetime.datetime.now().strftime('%Y-%m-%d')
    return str(now)


def to_binary(bars, threshold=0.0):
    """Turn velocity value into boolean"""
    track_is_max = tf.equal(bars, tf.reduce_max(bars, axis=-1, keepdims=True))
    track_pass_threshold = (bars > threshold)
    out_track = tf.logical_and(track_is_max, track_pass_threshold)
    return out_track

# Issue
# def save_midis(bars, file_path, tempo=80.0):
#     padded_bars = np.concatenate((np.zeros((bars.shape[0], bars.shape[1], 24, bars.shape[3])),
#                                   bars,
#                                   np.zeros((bars.shape[0], bars.shape[1], 20, bars.shape[3]))),
#                                  axis=2)
#     padded_bars = padded_bars.reshape(-1, 64, padded_bars.shape[2], padded_bars.shape[3])
#     padded_bars_list = []
#     for ch_idx in range(padded_bars.shape[3]):
#         padded_bars_list.append(padded_bars[:, :, :, ch_idx].reshape(padded_bars.shape[0],
#                                                                      padded_bars.shape[1],
#                                                                      padded_bars.shape[2]))
#     write_midi.write_piano_rolls_to_midi(piano_rolls=padded_bars_list,
#                                          program_nums=[0],
#                                          is_drum=[False],
#                                          filename=file_path,
#                                          tempo=tempo,
#                                          beat_resolution=4)

def abs_criterion(pred, target):
    return tf.reduce_mean(tf.abs(pred - target))


def mae_criterion(pred, target):
    return tf.reduce_mean((pred - target) ** 2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))


def softmax_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

#custom padding used to align datasets of different length
def padding(x, p=3):
    return tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")

#custom InstanceNorm class will allow Inctance normalization while training 
# 
class InstanceNorm(layers.Layer):
    def __init__(self, epsilon=1e-5):
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def call(self, x):
        scale = tf.Variable(
            initial_value=np.random.normal(1., 0.02, x.shape[-1:]),
            trainable=True,
            name='SCALE',
            dtype=tf.float32
        )
        offset = tf.Variable(
            initial_value=np.zeros(x.shape[-1:]),
            trainable=True,
            name='OFFSET',
            dtype=tf.float32
        )
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return scale * normalized + offset

#custom ResNet Layer defined
class ResNetBlock(layers.Layer):
    def __init__(self, dim, k_init, ks=3, s=1):
        super(ResNetBlock, self).__init__()
        self.dim = dim
        self.k_init = k_init
        self.ks = ks
        self.s = s
        self.p = (ks - 1) // 2
        # For ks = 3, p = 1
        self.padding = "valid"

    def call(self, x):
        y = layers.Lambda(padding, arguments={"p": self.p}, name="PADDING_1")(x)
        # After first padding, (batch * 130 * 130 * 3)

        y = layers.Conv2D(
            filters=self.dim,
            kernel_size=self.ks,
            strides=self.s,
            padding=self.padding,
            kernel_initializer=self.k_init,
            use_bias=False
        )(y)
        y = InstanceNorm()(y)
        y = layers.ReLU()(y)
        # After first conv2d, (batch * 128 * 128 * 3)

        y = layers.Lambda(padding, arguments={"p": self.p}, name="PADDING_2")(y)
        # After second padding, (batch * 130 * 130 * 3)

        y = layers.Conv2D(
            filters=self.dim,
            kernel_size=self.ks,
            strides=self.s,
            padding=self.padding,
            kernel_initializer=self.k_init,
            use_bias=False
        )(y)
        y = InstanceNorm()(y)
        y = layers.ReLU()(y + x)
        # After second conv2d, (batch * 128 * 128 * 3)

        return y

#creates discriminator:
#Structure: 3 convolutional layers of first two layers having 2 LeakyReLu activation
#Input: (batch * 64 * 84 * 1)
#Output:(batch * 16 * 21 * 1)
def build_discriminator(options, name='Discriminator'):

    initializer = tf.random_normal_initializer(0., 0.02)
    
    #input dimension: (batchsize, 64, 84, 1)
    input = Input(shape=(options.time_step,options.pitch_range,options.output_nc))

    x = input

    x = layers.Conv2D(filters=options.df_dim, kernel_size=7, strides=2, padding='same',
                    kernel_initializer=initializer, use_bias=False, name='CONV2D_1')(x)

    x = layers.LeakyReLU(alpha=0.2)(x)
    # (batch * 32 * 42 * 64)

    x = layers.Conv2D(filters=options.df_dim * 4,kernel_size=7,strides=2,padding='same',
                      kernel_initializer=initializer,use_bias=False,name='CONV2D_2')(x)

    x = InstanceNorm()(x)

    x = layers.LeakyReLU(alpha=0.2)(x)
    # (batch * 16 * 21 * 256)

    x = layers.Conv2D(filters=1, kernel_size=7, strides=1, padding='same',
                      kernel_initializer=initializer, use_bias=False, name='CONV2D_3')(x)
    # (batch * 16 * 21 * 1)

    output = x

    return Model(inputs=input, outputs=output, name=name)

#creates generator:
#Structure: 3 convolutional layers of first two layers, 10 ResNet layers, 
#           3 transposed convolutional layers, except for last one, all of them 
#           have activation function of ReLu
#Input: (batch * 64 * 84 * 1)
#Output:(batch * 64 * 84 * 1)
def build_generator(options, name='Generator'):
    initializer = tf.random_normal_initializer(0., 0.02)
    inputs = Input(shape=(options.time_step, options.pitch_range, options.output_nc))

    x = inputs

    x = layers.Lambda(padding, name='PADDING_1')(x)
    # (batch * 70 * 90 * 1)

    x = layers.Conv2D(filters=options.gf_dim, kernel_size=7, strides=1, padding='valid',
                      kernel_initializer=initializer, use_bias=False, name='CONV2D_1')(x)

    #Normalization
    x = InstanceNorm()(x)
    #Activation
    x = layers.ReLU()(x)
    # (batch * 64 * 84 * 64)

    x = layers.Conv2D(filters=options.gf_dim * 2, kernel_size=3, strides=2, padding='same',
                      kernel_initializer=initializer, use_bias=False, name='CONV2D_2')(x)

    x = InstanceNorm()(x)
    #Activation Function
    x = layers.ReLU()(x)
    # (batch * 32 * 42 * 128)

    x = layers.Conv2D(filters=options.gf_dim * 4, kernel_size=3, strides=2, padding='same',
                      kernel_initializer=initializer, use_bias=False, name='CONV2D_3')(x)

    #Normalization
    x = InstanceNorm()(x)
    #Activation Function
    x = layers.ReLU()(x)
    # (batch * 16 * 21 * 256)

    #10 ResNet Layer
    for i in range(10):
        # x = resnet_block(x, options.gf_dim * 4)
        x = ResNetBlock(dim=options.gf_dim * 4, k_init=initializer)(x)
    # (batch * 16 * 21 * 256)

    x = layers.Conv2DTranspose(filters=options.gf_dim * 2, kernel_size=3, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False, name='DECONV2D_1')(x)

    #Normalization
    x = InstanceNorm()(x)
    #Activation Function
    x = layers.ReLU()(x)
    # (batch * 32 * 42 * 128)

    x = layers.Conv2DTranspose(filters=options.gf_dim, kernel_size=3, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False, name='DECONV2D_2')(x)

    #Normalization      
    x = InstanceNorm()(x)
    #Activation Function
    x = layers.ReLU()(x)
    # (batch * 64 * 84 * 64)

    x = layers.Lambda(padding,name='PADDING_2')(x)
    # After padding, (batch * 70 * 90 * 64)

    x = layers.Conv2D(filters=options.output_nc, kernel_size=7, strides=1, padding='valid',
                      kernel_initializer=initializer, activation='sigmoid', use_bias=False, name='CONV2D_4')(x)
    # (batch * 64 * 84 * 1)

    outputs = x

    return Model(inputs=inputs, outputs=outputs, name=name)
                 
def build_discriminator_classifier(options, name='Discriminator_Classifier'):
    initializer = tf.random_normal_initializer(0., 0.02)
    inputs = Input(shape=(options.time_step,
                          options.pitch_range,
                          options.output_nc))

    x = inputs
    # (batch * 64, 84, 1)

    x = layers.Conv2D(filters=options.df_dim,
                      kernel_size=[1, 12],
                      strides=[1, 12],
                      padding='same',
                      kernel_initializer=initializer,
                      use_bias=False,
                      name='CONV2D_1')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    # (batch * 64 * 7 * 64)

    x = layers.Conv2D(filters=options.df_dim * 2,
                      kernel_size=[4, 1],
                      strides=[4, 1],
                      padding='same',
                      kernel_initializer=initializer,
                      use_bias=False,
                      name='CONV2D_2')(x)
    x = InstanceNorm()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    # (batch * 16 * 7 * 128)

    x = layers.Conv2D(filters=options.df_dim * 4,
                      kernel_size=[2, 1],
                      strides=[2, 1],
                      padding='same',
                      kernel_initializer=initializer,
                      use_bias=False,
                      name='CONV2D_3')(x)
    x = InstanceNorm()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    # (batch * 8 * 7 * 256)

    x = layers.Conv2D(filters=options.df_dim * 8,
                      kernel_size=[8, 1],
                      strides=[8, 1],
                      padding='same',
                      kernel_initializer=initializer,
                      use_bias=False,
                      name='CONV2D_4')(x)
    x = InstanceNorm()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    # (batch * 1 * 7 * 512)

    x = layers.Conv2D(filters=2,
                      kernel_size=[1, 7],
                      strides=[1, 7],
                      padding='same',
                      kernel_initializer=initializer,
                      use_bias=False,
                      name='CONV2D_5')(x)
    # (batch * 1 * 1 * 2)

    x = tf.reshape(x, [-1, 2])
    # (batch * 2)

    outputs = x

    return Model(inputs=inputs,
                 outputs=outputs,
                 name=name)
