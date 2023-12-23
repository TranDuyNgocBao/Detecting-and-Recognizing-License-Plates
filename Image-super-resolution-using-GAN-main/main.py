# -*- coding: utf-8 -*-

from model import SRGAN
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
import dataset as data
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

srgan = SRGAN()

# Load data
lr_size = (24, 24)
hr_size = (lr_size[0]*4, lr_size[1]*4)
hr, lr = data.load_data_train('E:\Subjects\AIP391\mirflickr25k\mirflickr\mirflickr',lr_size)

print(type(hr))
print(type(lr))

import random
id = random.randint(0, len(hr))
plt.subplot(121)
plt.imshow(hr[id].astype(np.uint8))

plt.subplot(122)
plt.imshow(lr[id].astype(np.uint8))
plt.show()

def train(hr_train, lr_train, save_path, lr_size, batch_size=16, epochs=5):
    if not os.path.exists(save_path + '/{}_{}'.format(lr_size, (lr_size[0] * 4, lr_size[1] * 4))):
        os.makedirs(save_path + '/{}_{}'.format(lr_size, (lr_size[0] * 4, lr_size[1] * 4)))

    print("Done loading training datasets............")
    hr_train = hr_train / 127.5 - 1
    lr_train = lr_train / 127.5 - 1


    hr_shape = hr_train.shape
    lr_shape = lr_train.shape
    print("HR shape:", hr_shape)
    print("LR shape:", lr_shape)
    
    hr_input = Input(shape=(hr_shape[1], hr_shape[2], hr_shape[3]))
    lr_input = Input(shape=(lr_shape[1], lr_shape[2], lr_shape[3]))
    generator = srgan.generator(lr_input)
    
    # generator.summary()

    discriminator = srgan.discriminator(hr_input)
    discriminator.compile(loss="binary_crossentropy",
                          optimizer=Adam(learning_rate=0.0001),
                          metrics=['accuracy'])
    # discriminator.summary()

    vgg = srgan.build_vgg((hr_shape[1], hr_shape[2], hr_shape[3]))
    # vgg = srgan.build_vgg((96, 96, 3))
    vgg.trainable = False
    

    gan_model = srgan.gan(generator, discriminator, vgg, lr_input, hr_input)
    gan_model.compile(loss=["binary_crossentropy", "mse"],
                      loss_weights=[1e-3, 1],
                      optimizer=Adam(learning_rate=0.0001))
    train_hr_batches, train_lr_batches = data.batch_data(hr_shape[0], hr_train, lr_train,
                                                         batch_size)

    g = []
    d = []
    print("Training............")
    for e in range(epochs):
        fake_label = np.zeros((batch_size, 1))  # Assign a label of 0 to all fake (generated images)
        real_label = np.ones((batch_size, 1))  # Assign a label of 1 to all real images.

        g_losses = []
        d_losses = []

        for b in tqdm(range(len(train_hr_batches))):
            lr_imgs = train_lr_batches[b]  # Fetch a batch of LR images for training
            hr_imgs = train_hr_batches[b]  # Fetch a batch of HR images for training

            fake_imgs = generator.predict_on_batch(lr_imgs)  # Fake images
            # print(fake_imgs)
            # First, train the discriminator on fake and real HR images.
            discriminator.trainable = True
            d_loss_gen, _ = discriminator.train_on_batch(fake_imgs, fake_label)
            d_loss_real, _ = discriminator.train_on_batch(hr_imgs, real_label)

            # Now, train the generator by fixing discriminator as non-trainable
            discriminator.trainable = False

            # Average the discriminator loss, just for reporting purposes.
            d_loss = 0.5 * np.add(d_loss_gen, d_loss_real)

            # Extract VGG features, to be used towards calculating loss
            image_features = vgg.predict(hr_imgs)

            # Train the generator via GAN.
            # Remember that we have 2 losses, adversarial loss and content (VGG) loss
            g_loss, a, b = gan_model.train_on_batch([lr_imgs, hr_imgs], [real_label, image_features])
            # print(g_loss, a, b)
            # Save losses to a list so we can average and report.
            d_losses.append(d_loss)
            g_losses.append(g_loss)

        # Convert the list of losses to an array to make it easy to average
        g_losses = np.array(g_losses)
        d_losses = np.array(d_losses)

        # Calculate the average losses for generator and discriminator
        g_loss = np.sum(g_losses, axis=0) / len(g_losses)
        d_loss = np.sum(d_losses, axis=0) / len(d_losses)

        g.append(g_loss)
        d.append(d_loss)
        # Report the progress during training.

        # generator.save("gen_abc/e_" + str(e + 1) + ".h5")
        generator.save_weights(
            save_path + '/{}_{}/e_{}.h5'.format(lr_size, (lr_size[0] * 4, lr_size[1] * 4), e + 1))
        # psnr = validation(e + 1)
        print("epoch {}:  g_loss: {} - d_loss: {}".format(e + 1, g_loss, d_loss))
    print()
    return g, d

# hr_train = hr_train / 127.5 - 1
# lr_train = lr_train / 127.5 - 1
g, d = train(hr, lr, 
             'E:\Subjects\AIP391\Image-super-resolution-using-GAN-main\weight',
             lr_size,
             epochs=5)

plt.plot(range(1, 5+1), g, label='generator')
plt.plot(range(1, 5+1), d, label='discriminator')
plt.legend()
plt.show()

# weight = '/content/drive/MyDrive/AIP391/Weight_generation/(24, 24)_(96, 96)/e_50_sigmoid.h5'
weight = 'E:\Subjects\AIP391\Image-super-resolution-using-GAN-main\weight\e_77.h5'
def validation(hr_valid_path, lr_valid_path, weight_path):
    generator = srgan.generator(Input(shape=(None, None, 3)))
    generator.load_weights(weight_path)
    hr_valid, lr_valid = data.load_data_val(hr_valid_path, lr_valid_path)

    hr_valid[0] = hr_valid[0] / 127.5 - 1
    lr_valid[0] = lr_valid[0] / 127.5 - 1
    # lr_valid /= 255.

    print('HR shape:', hr_valid.shape)
    print('LR shape:', lr_valid.shape)

    sr_img = generator.predict(lr_valid)
    
    hr = np.asarray((hr_valid[0] + 1) * 127.5, dtype=np.uint8)
    lr = np.asarray((lr_valid[0] + 1) * 127.5, dtype=np.uint8)
    sr = np.asarray((sr_img[0] + 1) * 127.5, dtype=np.uint8)
    # sr = np.asarray(sr_img[0] * 255., dtype=np.uint8)
    print(cv2.PSNR(sr, hr))
    plt.figure(figsize=(15, 15))
    plt.subplot(131)
    plt.imshow(lr)
    # plt.imshow(lr_valid[0])
    plt.title("LR")
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(sr)
    plt.title("SR")
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(hr)
    plt.title("HR")
    plt.axis('off')

    plt.show()

validation('E:\Subjects\AIP391\mirflickr25k\hr_images\im1.jpg',
          'E:\Subjects\AIP391\mirflickr25k\lr_images\im1.jpg',
           weight)