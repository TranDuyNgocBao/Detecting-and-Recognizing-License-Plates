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
device_selected = select_device('') # choose cuda or cpu
model = attempt_load(opt['weights'], map_location=device_selected)
model_2 = attempt_load(opt_2['weights'], map_location=device_selected)
#Recognize
reader = easyocr.Reader(['en'], gpu=True)
#SRGAN
generator.load_weights("weight_touse/Gen_120.h5")

# PAGE
st.set_page_config(
    page_title="Home",
    page_icon="H"
)

st.title("Detecting and Recognizing License Plates")
st.sidebar.success("Select a page above")

#LOAD Variables
if "opt" not in st.session_state:
    st.session_state['opt'] = opt

if "opt_2" not in st.session_state:
    st.session_state['opt_2'] = opt_2

if "device_selected" not in st.session_state:
    st.session_state['device_selected'] = device_selected

if "model" not in st.session_state:
    st.session_state['model'] = model

if "model_2" not in st.session_state:
    st.session_state['model_2'] = model_2

if "reader" not in st.session_state:
    st.session_state['reader'] = reader

if "generator" not in st.session_state:
    st.session_state["generator"] = generator


