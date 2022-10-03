import streamlit as st
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

def singleImage(img_r, nightmode = False):
    # nightmode
    if nightmode:
        # img_r = ZeROdce_net.test(img_r, zero_model)

        tmp_img_r = lowlight(img_r)  # PYTORCH
        img_r = tmp_img_r[:, :, ::-1].copy()  # PYTORCH

    # Detect
    img0, pred_pl = detect_image(st.session_state['opt_2'], img_r, st.session_state['model_2'], st.session_state['device_selected'])
    img0, pred_vh = detect_image(st.session_state['opt'], img0, st.session_state['model'], st.session_state['device_selected'])

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
                    new_img = SRGAN_mode.test(cropped_frame, st.session_state['generator'], False)
                    text = img_to_text(st.session_state['reader'], new_img)
                else:
                    text = img_to_text(st.session_state['reader'], cropped_frame)

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

            cv2.imwrite("output_videos/" + 'save.jpg', img0)
            st.subheader(f"{j + 1}/{nframes} frames")
            st.image("output_videos/" + 'save.jpg')
            os.remove("output_videos/" + 'save.jpg')

            if recording:
                output.write(img0)
        else:
            break

    if recording:
        output.release()
        video.release()

    return nameOfvideo

st.title("Detect and Recognize Video")

st.header("Chose your video from PC")

night = False
button_night = st.button("NightMode")
if button_night:
    night = True

if night:
    st.write("NightMode ON")
else:
    st.write("NightMode OFF")

video = st.file_uploader("Upload video:", type=["mp4", "mpeg"])

if video is not None:
    st.subheader("Orginal Video")
    st.video(video)

    name_video = videoReg("input_videos/" + video.name, night)
    st.subheader("Processed Video")
    st.video("output_videos/" + name_video + ".mp4")
    os.remove("output_videos/" + name_video + ".mp4")