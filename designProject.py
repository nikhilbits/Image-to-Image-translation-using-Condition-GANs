import os
import datetime
import imageio
import skimage
import scipy # 

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from glob import glob
from IPython.display import Image

tf.logging.set_verbosity(tf.logging.ERROR)

class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"
        path = glob('../input/%s/%s/%s/*' % (self.dataset_name, self.dataset_name, data_type))
        
        batch_images = np.random.choice(path, size=batch_size)

        imgs_A = []
        imgs_B = []
        for img_path in batch_images:
            img = self.imread(img_path)

            h, w, _ = img.shape
            _w = int(w/2)
            img_A, img_B = img[:, :_w, :], img[:, _w:, :]

            img_A = scipy.misc.imresize(img_A, self.img_res)
            img_B = scipy.misc.imresize(img_B, self.img_res)

            # If training => do random flip
            #if not is_testing and np.random.random() < 0.5:
             #   img_A = np.fliplr(img_A)
              #  img_B = np.fliplr(img_B)

            imgs_A.append(img_A)
            imgs_B.append(img_B)

        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.

        return imgs_A, imgs_B

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"
        path = glob('../input/%s/%s/%s/*' % (self.dataset_name, self.dataset_name, data_type))

        self.n_batches = int(len(path) / batch_size)

        for i in range(self.n_batches-1):
            batch = path[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img in batch:
                img = self.imread(img)
                h, w, _ = img.shape
                half_w = int(w/2)
                img_A = img[:, :half_w, :]
                img_B = img[:, half_w:, :]

                img_A = scipy.misc.imresize(img_A, self.img_res)
                img_B = scipy.misc.imresize(img_B, self.img_res)

               # if not is_testing and np.random.random() > 0.5:
                #        img_A = np.fliplr(img_A)
                 #       img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B


    def imread(self, path):
        return imageio.imread(path).astype(np.float)


class Pix2Pix():
    def __init__(self):
        # Input shape
        self.imgRows = 256
        self.imgCols = 256
        self.channels = 3
        self.imgShape = (self.imgRows, self.imgCols, self.channels)

        # Configure data loader
        self.datasetName = 'facades'
        self.dataLoader = DataLoader(dataset_name=self.datasetName, img_res=(self.imgRows, self.imgCols))


        # Calculate output shape of D (PatchGAN)
        patch = 70
        self.discPatch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        #self.gf = 64
        #self.df = 64

        optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.buildDiscriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator = self.buildGenerator()

        # Input images and their conditioning images
        imgA = tf.keras.layers.Input(shape=self.imgShape)
        imgB = tf.keras.layers.Input(shape=self.imgShape)

        # By conditioning on B generate a fake version of A
        fakeA = self.generator(imgB)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fakeA, imgB])

        self.combined = tf.keras.models.Model(inputs=[imgA, imgB], outputs=[valid, fakeA])
        self.combined.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)

    def buildGenerator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True, dropout_rate=0):
            """Layers used during downsampling"""
            d = tf.keras.layers.Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
            if dropout_rate:
                d = tf.keras.layers.Dropout(dropout_rate)(d)
            if bn:
                d = tf.keras.layers.BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = tf.keras.layers.UpSampling2D(size=2)(layer_input)
            u = tf.keras.layers.Conv2D(filters, kernel_size=f_size, strides=2, padding='same', activation='relu')(u)
            if dropout_rate:
                u = tf.keras.layers.Dropout(dropout_rate)(u)
            u = tf.keras.layers.BatchNormalization(momentum=0.8)(u)
            u = tf.keras.layers.Concatenate()([u, skip_input])
            return u

        # Image input
        downSample0 = tf.keras.layers.Input(shape=self.imgShape)

        # Downsampling
        downSample1 = conv2d(downSample0, 64, bn=False)
        downSample2 = conv2d(downSample1, 128)
        downSample3 = conv2d(downSample2, 256)
        downSample4 = conv2d(downSample3, 512)
        downSample5 = conv2d(downSample4, 512)
        downSample6 = conv2d(downSample5, 512)
        downSample7 = conv2d(downSample6, 512)
     	downSample8 = conv2d(downSample7, 512)
        # Upsampling
        upSample1 = deconv2d(downSample8, downSample7, 1024, dropout_rate = 0.5)
        upSample2 = deconv2d(downSample8, downSample6, 1024, dropout_rate = 0.5)
        upSample3 = deconv2d(upSample2, downSample5, 1024, dropout_rate = 0.5)
        upSample4 = deconv2d(upSample3, downSample4, 1024)
        upSample5 = deconv2d(upSample4, downSample3, 512)
        upSample6 = deconv2d(upSample5, downSample2, 256)
        upSample7 = deconv2d(upSample6, downSample1, 128)
        upSample8 = tf.keras.layers.UpSampling2D(size=2)(upSample7)
        outputImg = tf.keras.layers.Conv2D(self.channels, kernel_size=4, strides=2, padding='same', activation='tanh')(upSample8)

        return tf.keras.models.Model(downSample0, outputImg)

    def buildDiscriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = tf.keras.layers.Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
            if bn:
                d = tf.keras.layers.BatchNormalization(momentum=0.8)(d)
            return d

        imgA = tf.keras.layers.Input(shape=self.imgShape)
        imgB = tf.keras.layers.Input(shape=self.imgShape)

        # Concatenate image and conditioning image by channels to produce input
        combinedImgs = tf.keras.layers.Concatenate(axis=-1)([imgA, imgB])

        downSample1 = d_layer(combinedImgs, 64, bn=False)
        downSample2 = d_layer(downSample1, 128)
        downSample3 = d_layer(downSample2, 256)
        downSample4 = d_layer(downSample3, 512)

        validity = tf.keras.layers.Conv2D(1, kernel_size=4, strides=2, padding='same')(downSample4)

        return tf.keras.models.Model([imgA, imgB], validity)

    def train(self, epochs, batch_size=1, sample_interval=50):
        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.discPatch)
        fake = np.zeros((batch_size,) + self.discPatch)

        for epoch in range(epochs):
            for batch_i, (imgsA, imgsB) in enumerate(self.data_loader.load_batch(batch_size)):
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                fakeA = self.generator.predict(imgsB)

                # Train the discriminators (original images = real / generated = Fake)
                dLossReal = self.discriminator.train_on_batch([imgsA, imgsB], valid)
                dLossFake = self.discriminator.train_on_batch([fakeA, imgsB], fake)
                dLoss = 0.5 * np.add(dLossReal, dLossFake)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                gLoss = self.combined.train_on_batch([imgsA, imgsB], [valid, imgsA])

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                        batch_i, self.data_loader.n_batches,
                                                                        dLoss[0], 100*dLoss[1],
                                                                        gLoss[0],
                                                                        elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sampleImages(epoch, batch_i)

    def sampleImages(self, epoch, batch_i):
        os.makedirs('./images/%s' % self.dataset_name, exist_ok=True)
        r, c = 3, 3

        imgsA, imgsB = self.data_loader.load_data(batch_size=3, is_testing=True)
        fakeA = self.generator.predict(imgsB)

        genImgs = np.concatenate([imgsB, fakeA, imgsA])

        # Rescale images 0 - 1
        genImgs = 0.5 * genImgs + 0.5

        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[i])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("./images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        plt.close()
