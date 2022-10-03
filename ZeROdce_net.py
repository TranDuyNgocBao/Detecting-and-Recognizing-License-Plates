from model_dce import ZeroDCE
from PIL import Image
import config_dce
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

# zero_model = ZeroDCE(shape=(None, None, 3))
# zero_model.compile(learning_rate=1e-4)

def test(original_image, zero_model):
    # original_image = Image.open(original_image)
    # image = img_to_array(original_image)
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    output_image = zero_model(image)
    output_image = tf.cast((output_image[0, :, :, :] * 255), dtype=np.uint8)
    # output_image = Image.fromarray(output_image.numpy())
    output_image = output_image.numpy()

    # print(output_image)

    return output_image

if __name__ == '__main__':
    source_image_path = 'test_images/59S228834.jpg'
    img_r = cv2.imread(source_image_path)

    # print(img_r.shape)

    out_img = test(img_r)

    # print(out_img.numpy().shape)

    plt.subplot(121)
    plt.imshow(out_img)
    plt.subplot(122)
    plt.imshow(img_r)
    plt.show()