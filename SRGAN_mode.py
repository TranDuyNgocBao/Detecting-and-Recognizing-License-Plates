from modelofSrgan import SRGAN
from tensorflow.keras.layers import Input
import datasetSRgan as data
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# srgan = SRGAN()
# generator = srgan.generator(Input(shape=(None, None, 3)))
# weight = "weight_touse/Gen_120.h5"
# generator.load_weights(weight)

def test(lr_test_path, generator, cuda_env = True):
    # lr_test = data.load_data_test(lr_test_path)
    # lr_test = cv2.imread(lr_test_path)
    if cuda_env is False:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    lr_test = cv2.cvtColor(lr_test_path, cv2.COLOR_BGR2RGB)
    lr_test = np.expand_dims(lr_test, axis=0)

    lr_test = np.array(lr_test, dtype=float)

    lr_test[0] = lr_test[0] / 127.5 - 1

    # print('LR shape:', lr_test.shape)
    sr_img = generator.predict(lr_test)
    # print('SR shape:', sr_img.shape)

    # lr = np.asarray((lr_test[0] + 1) * 127.5, dtype=np.uint8)
    sr = np.asarray((sr_img[0] + 1) * 127.5, dtype=np.uint8)

    # plt.subplot(121)
    # plt.imshow(lr)
    # plt.title("LR image")
    # plt.axis('off')
    #
    # plt.subplot(122)
    # plt.imshow(sr)
    # plt.title("SR image")
    # plt.axis('off')
    #
    # plt.show()
    # if save:
    #     cv2.imwrite('result/lr_test.png', lr[:, :, ::-1])
    #     cv2.imwrite('result/sr_test.png', sr[:, :, ::-1])

    # print(sr.shape, type(sr))

    return sr

if __name__ == '__main__':
    choice = 3

    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # if choice == 1:
    #     train('Dataset/DIV2K_train_HR',
    #           'Weight_generation',
    #           (24, 24))
    # elif choice == 2:
    #     validation('Dataset/Urban100_SR/image_SRF_4/img_005_SRF_4_HR.png',
    #                'Dataset/Urban100_SR/image_SRF_4/img_005_SRF_4_LR.png',
    #                weight)
    # if choice == 3:
    #     test(cv2.imread('test_images/Screenshot 2022-09-25 155701.png'),
    #          weight)

    # elif choice == 4:
    #     test_video("test video/origin/HR2",
    #                "test video/LR_video/LR2.mp4",
    #                "test video/Enhance/SR2.mp4",
    #                weight, scale=10)