import os
from flask import Flask, render_template, send_from_directory, url_for
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
import base64
import numpy as np
import cv2
from keras.models import load_model

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

app = Flask(__name__)
app.config['SECRET_KEY'] = 'lam'
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'
app.config['ENHANCED_IMAGE'] = 'enhance'

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)


def SRGAN():
    loaded_model = load_model('E:\Subjects\AIP391\image-super-resolution-web-app-main\model\generator.h5', compile=False)
    loaded_model.load_weights('E:\Subjects\AIP391\image-super-resolution-web-app-main\model\e_50_127.5_127.5.h5')
    return loaded_model


def image_to_bas64(image_name):
    with open(image_name[1:], 'rb') as img_file:
        b64 = base64.b64encode(img_file.read())
    return b64


def enhance_image(image, name):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.asarray(image, dtype=float)
    image = image / 127.5 - 1
    image = np.expand_dims(image, axis=0)
    model = SRGAN()
    sr_img = model.predict(image)
    sr = np.asarray((sr_img[0] + 1) * 127.5, dtype=np.uint8)
    cv2.imwrite(name[1:], sr[:, :, ::-1])
    return sr


def base64_to_array(b64):
    img_bytes = base64.b64decode(b64)
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_arr, flags=cv2.IMREAD_COLOR)
    return img


class UploadForm(FlaskForm):
    photo = FileField(
        validators=[
            FileAllowed(photos, 'Only images are allowed'),
            FileRequired('File field should not to be empty')
        ]
    )
    submit = SubmitField('Enhance')


@app.route('/uploads/<filename>')
def get_upload_file(filename):
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'], filename)


@app.route('/enhance/<filename>')
def get_enhance_file(filename):
    return send_from_directory(app.config['ENHANCED_IMAGE'], filename)


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    form = UploadForm()
    size_message = None
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = url_for('get_upload_file', filename=filename)
        img_tmp = cv2.imread(file_url[1:])
        img_tmp_shape = img_tmp.shape
        if img_tmp_shape[0] > 500 or img_tmp_shape[1] > 500:
            lr_link = None
            sr_link = None
            size_message = 'Please input weight or height of image <= 500px'
        else:
            lr_bs64 = image_to_bas64(file_url)
            _, image_extension = os.path.splitext(file_url)
            lr_link = 'data:image/{};base64, {}'.format(image_extension[1:], lr_bs64.decode('utf-8'))

            lr_image = base64_to_array(lr_bs64)

            sr_path = url_for('get_enhance_file', filename=filename)
            enhance_image(lr_image, sr_path)

            sr_bs64 = image_to_bas64(sr_path)
            sr_link = 'data:image/{};base64, {}'.format(image_extension[1:], sr_bs64.decode('utf-8'))
    else:
        lr_link = None
        sr_link = None
    return render_template('index.html', size_message=size_message, form=form, lr_link=lr_link, sr_link=sr_link)


if __name__ == '__main__':
    app.run()

