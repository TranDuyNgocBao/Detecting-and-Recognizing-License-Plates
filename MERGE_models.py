import argparse
import time
from pathlib import Path
import cv2
import torch
torch.cuda.empty_cache()
import numpy as np
import torch.backends.cudnn as cudnn
from numpy import random
import easyocr
import pandas as pd
import ZeROdce_net
from dce_pytorch.lowlight_test import lowlight #PYTORCH
import config_dce
import SRGAN_mode
import os, os.path
import matplotlib.pyplot as plt

from models.experimental import attempt_load
from tensorflow.keras.layers import Input
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from model_dce import ZeroDCE
from modelofSrgan import SRGAN


#Setting parameters
# setting opt
classes_to_filter = ['car', 'motorcycle', 'bus', 'truck']
opt = {

    "weights": "weight_touse/yolov7.pt",  # Path to weights file default weights are for nano model
    "yaml": "YOLOV7\data\coco.yaml",
    "img-size": 640,  # default image size
    "conf-thres": 0.25,  # confidence threshold for inference.
    "iou-thres": 0.45,  # NMS IoU threshold for inference.
    "device": '',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
    "classes": classes_to_filter  # list of classes to filter or None

}

classes_to_filter_2 = ['plate']
opt_2 = {

    "weights": "weight_touse/detected.pt",
    # Path to weights file default weights are for nano model
    "yaml": "YOLOV7\data\mydataset.yaml",
    "img-size": 640,  # default image size
    "conf-thres": 0.25,  # confidence threshold for inference.
    "iou-thres": 0.45,  # NMS IoU threshold for inference.
    "device": '',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
    "classes": classes_to_filter_2  # list of classes to filter or None

}

# Setting weight DCE
zero_model = ZeroDCE(shape=(None, None, 3))
zero_model.compile(learning_rate=1e-4)

# Setting weight SRGAN
srgan = SRGAN()
generator = srgan.generator(Input(shape=(None, None, 3)))

# Load models
#Detect
device_selected = select_device('cpu') # choose cuda or cpu
model = attempt_load(opt['weights'], map_location=device_selected)
model_2 = attempt_load(opt_2['weights'], map_location=device_selected)
#Recognize
reader = easyocr.Reader(['en'], gpu=False)
#DCE
zero_model.load_weights(config_dce.SAVE_WEIGHT_PATH)
#SRGAN
generator.load_weights("weight_touse/Gen_120.h5")



def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def get_name_video(path_to_output):

  # path joining version for other paths
  DIR = path_to_output
  temp = str(len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]))

  video_save = path_to_output + '/' + temp
  return video_save

def detect_image(opt, img_r, model, device):
    with torch.no_grad():
        weights, imgsz = opt['weights'], opt['img-size']
        set_logging()
        # device = select_device(opt['device'])
        half = device.type != 'cpu'
        # model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if half:
            model.half()

        names = model.module.names if hasattr(model, 'module') else model.names
        random.seed(42)
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

        img0 = img_r
        img = letterbox(img0, imgsz, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=False)[0]

        # Apply NMS
        num_classes = None
        if opt['classes']:
            num_classes = []
            for class_name in opt['classes']:
                num_classes.append(names.index(class_name))

        if num_classes == None:
            num_classes = [i for i in range(len(names)) if i not in num_classes]

        pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'], classes=num_classes, agnostic=True)
        t2 = time_synchronized()
        for i, det in enumerate(pred):
            s = ''
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
            if len(det):

                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, img0, color=colors[int(cls)], line_thickness=2)  # , label=label
        return img0, pred

def img_to_text(reader, img):
    result = reader.readtext(img)

    extracts = pd.DataFrame(result, columns=['bbox', 'text', 'conf'])

    text = ''
    for strs in extracts['text']:
        text += strs + ' '

    if len(text) < 7:
        return 'Cannot recognize plates'

    return text

def display_video(path_to_video):
    cap = cv2.VideoCapture(path_to_video)

    while(cap.isOpened()):
        ret, frame = cap.read()

        frame = cv2.resize(frame, (640, 640))

        cv2.imshow("video", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def singleImage(img_r, nightmode = False):
    # nightmode
    if nightmode:
        # img_r = ZeROdce_net.test(img_r, zero_model)

        tmp_img_r = lowlight(img_r)  # PYTORCH
        img_r = tmp_img_r[:, :, ::-1].copy()  # PYTORCH

    # Detect
    img0, pred_pl = detect_image(opt_2, img_r, model_2, device_selected)
    img0, pred_vh = detect_image(opt, img0, model, device_selected)

    # cv2.imshow('', img0)
    # cv2.waitKey(0)

    # torch.cuda.empty_cache()

    #Check active
    verify_act = False

    # Recognize
    for i, det in enumerate(pred_pl):
        if len(det):
            verify_act = True

            for *xyxy, conf, cls in reversed(det):
                x1 = int(xyxy[0].item())
                y1 = int(xyxy[1].item())
                x2 = int(xyxy[2].item())
                y2 = int(xyxy[3].item())

                confidence_score = conf
                class_index = cls

                # print('bouding box is ', x1, y1, x2, y2)
                # print('class index is ', class_index)

                original_image = img0
                cropped_frame = original_image[y1:y2, x1:x2]
                # print(type(cropped_frame))
                if len(cropped_frame) <= 450 and len(cropped_frame[0]) <= 450:
                    # print('Im here')
                    new_img = SRGAN_mode.test(cropped_frame, generator, False)
                    text = img_to_text(reader, new_img)
                else:
                    text = img_to_text(reader, cropped_frame)

                cv2.putText(img0, text,(x1+3, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60,255,255),2)

    return img0, verify_act

def videoReg(video_path, nightmode):
    video = cv2.VideoCapture(video_path, 0)

    nameOfvideo = get_name_video('output_videos') + '.mp4'
    recording = False

    # Video information
    fps = video.get(cv2.CAP_PROP_FPS)
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialzing object for writing video output
    if recording is False:
        output = cv2.VideoWriter(nameOfvideo, cv2.VideoWriter_fourcc(*'DIVX'), fps, (w, h))
        torch.cuda.empty_cache()
        recording = True

    for j in range(nframes):
        ret, img0 = video.read()

        cv2.putText(img0, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if ret:
            img0, ver = singleImage(img0, nightmode)



            print(f"{j + 1}/{nframes} frames processed")

            if recording:
                output.write(img0)
        else:
            break

    if recording:
        output.release()
        video.release()

    return nameOfvideo

def video_stream(nightmode):
    cap = cv2.VideoCapture(0)

    # Calculate FPS
    fps_start_time = 0
    fps = 0

    while True:
        ret, frame = cap.read()

        frame, verify_act = singleImage(frame, nightmode)
        if verify_act:
            cv2.imwrite(get_name_video('output_from_stream'), frame)

        fps_end_time = time.time()
        time_diff = fps_end_time - fps_start_time
        fps = 1 / (time_diff)
        print("fps: ", "{:.2f}".format(fps))
        fps_start_time = fps_end_time
        fps_text = ".2f".format(fps)
        cv2.putText(frame, "{:.2f}".format(fps), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)

        cv2.imshow('streaming', frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    nightmode = False
    while True:
        funct = input('Choose function: ')
        if funct == "1":
            name_im = cv2.imread('test_images/'+input('Image name: '))
            img, ver = singleImage(name_im, nightmode)
            cv2.imshow('image',img)
            cv2.waitKey(0)

        elif funct == "2":
            name_vid = 'input_videos/' + input('Name video: ')
            name_provid = videoReg(name_vid, nightmode)
            display_video(name_provid)

        elif funct == "3":
            video_stream(nightmode)

        else:
            break