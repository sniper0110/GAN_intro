import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import LeakyReLU, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt
import os

class Discriminator:

    def __init__(self):
        self.img_w = 28
        self.img_h = 28
        self.input_shape = (self.img_h, self.img_w, 1)
        self.dropout_rate = 0.5
        self.net = None

        self.create_network()

    def create_network(self):

        self.input = Input(shape=self.input_shape, name='to_this_link')

        self.conv_1 = Conv2D(filters=64, kernel_size=[3, 3], strides=(1, 1), padding='same', activation='relu')(self.input)
        #print("self.conv_1.output_shape = ", self.conv_1.output_shape)
        self.dropout_1 = Dropout(self.dropout_rate)(self.conv_1)
        self.conv_2 = Conv2D(64 * 2, 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2))(self.dropout_1)
        self.dropout_2 = Dropout(self.dropout_rate)(self.conv_2)
        self.conv_3 = Conv2D(64 * 4, 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2))(self.dropout_2)
        self.dropout_3 = Dropout(self.dropout_rate)(self.conv_3)
        self.conv_4 = Conv2D(64 * 8, 5, strides=1, padding='same', activation=LeakyReLU(alpha=0.2))(self.dropout_3)
        self.dropout_4 = Dropout(self.dropout_rate)(self.conv_4)
        self.flatten_1 = Flatten()(self.dropout_4)

        self.fc_1 = Dense(1, activation='sigmoid', name='output_discriminator')(self.flatten_1)

        self.net = Model(inputs=self.input, outputs=self.fc_1, name="Discriminator")

        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.net.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


class Generator:

    def __init__(self):

        self.input_shape = (100,)
        self.dropout_rate = 0.5
        self.net = None
        self.create_network()

    def create_network(self):

        depth = 64 + 64 + 64 + 64
        dim = 7
        self.input = Input(shape=self.input_shape, name='input_generator')
        self.fc_1 = Dense(7 * 7 * depth, input_dim=100)(self.input)

        #print("self.fc_1.output_shape = ", self.fc_1.output_shape)
        self.batch_norm_1 = BatchNormalization(momentum=0.9)(self.fc_1)
        self.activation_1 = Activation('relu')(self.batch_norm_1)
        self.reshape_1 = Reshape((dim, dim, depth))(self.activation_1)
        self.dropout_1 = Dropout(self.dropout_rate)(self.reshape_1)
        self.upsample_1 = UpSampling2D()(self.dropout_1)
        self.conv_1 = Conv2DTranspose(int(depth / 2), 5, padding='same')(self.upsample_1)
        self.batch_norm_2 = BatchNormalization(momentum=0.9)(self.conv_1)
        self.activation_2 = Activation('relu')(self.batch_norm_2)
        self.upsample_2 = UpSampling2D()(self.activation_2)
        self.t_conv_1 = Conv2DTranspose(int(depth / 4), 5, padding='same')(self.upsample_2)
        self.batch_norm_3 = BatchNormalization(momentum=0.9)(self.t_conv_1)
        self.activation_3 = Activation('relu')(self.batch_norm_3)
        self.t_conv_2 = Conv2DTranspose(int(depth / 8), 5, padding='same')(self.activation_3)
        self.batch_norm_4 = BatchNormalization(momentum=0.9)(self.t_conv_2)
        self.activation_4 = Activation('relu')(self.batch_norm_4)
        self.t_conv_3 = Conv2DTranspose(1, 5, padding='same')(self.activation_4)

        self.activation_5 = Activation('sigmoid', name='link_this')(self.t_conv_3)

        self.net = Model(inputs=self.input, outputs=self.activation_5, name="Generator")




class AdverserialModel:

    def __init__(self):

        self.img_rows = 28
        self.img_cols = 28
        self.channel = 1

        self.x_train = input_data.read_data_sets("mnist", one_hot=True).train.images
        print("self.x_train.shape = ", self.x_train.shape)
        self.x_train = self.x_train.reshape(-1, self.img_rows, self.img_cols, 1).astype(np.float32)
        print("self.x_train.shape (after reshape) = ", self.x_train.shape)
        self.input_shape = 100
        self.discriminator = Discriminator()
        self.generator = Generator()
        self.net = None
        self.create_adversarial_model()


    def create_adversarial_model(self):

        optimizer = RMSprop(lr=0.0001, decay=3e-8)

        self.inputs = self.generator.net.input
        #self.discriminator.net.input = self.generator.net.output
        self.outputs = self.discriminator.net(self.generator.net.output)
        self.net = Model(inputs=self.inputs, outputs=self.outputs, name="Adverserial")

        print("self.net.input_shape = ", self.net.input_shape)
        print("self.net.output_shape = ", self.net.output_shape)
        """
        https://github.com/keras-team/keras/issues/4205#issuecomment-257284099
        """

        """
        self.net = Sequential()
        self.net.add(self.generator.net)
        self.net.add(self.discriminator.net)
        """
        print(self.net.summary())
        #plot_model(self.net, to_file=r"C:\Users\islam\Desktop\gan_model.png")
        self.net.compile(loss='binary_crossentropy', optimizer=optimizer) #, metrics=['accuracy']


    def train(self, train_steps=2000, batch_size=256, save_interval=50):

        noise_input = None

        if save_interval > 0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])

        for i in range(train_steps):

            images_train = self.x_train[np.random.randint(0, self.x_train.shape[0], size=batch_size), :, :, :]
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            images_fake = self.generator.net.predict(noise)
            x = np.concatenate((images_train, images_fake))
            y = np.ones([2 * batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = self.discriminator.net.train_on_batch(x, y)

            y = np.ones([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            a_loss = self.net.train_on_batch(noise, y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f]" % (log_mesg, a_loss) # [0], a_loss[1]
            print(log_mesg)


            if save_interval > 0:
                if (i + 1) % save_interval == 0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0], noise=noise_input, step=(i + 1))


    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = 'mnist.png'
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
            else:
                filename = os.path.join("output_images", "mnist_%d.png" % step)
            images = self.generator.net.predict(noise)
        else:
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]

        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()

if __name__ == '__main__':

    adverserial_net = AdverserialModel()
    adverserial_net.train()