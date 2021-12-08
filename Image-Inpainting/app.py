import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'lama'))
sys.path.append(os.path.join(os.getcwd(), 'modules'))
import cv2
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate


from lama.saicinpainting.training.trainers import load_checkpoint
from modules.inference import inference
from modules.utils import button

import streamlit as st
from streamlit_drawable_canvas import st_canvas


import PIL.Image as Image
import time
import albumentations as albu
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from iglovikov_helper_functions.utils.image_utils import pad, unpad
from people_segmentation.pre_trained_models import create_model

device = torch.device('cuda')
path = '/opt/ml/data/ImageInpainting/lama/LaMa_models/big-lama' #FIXME

@st.cache
def load_model():
    # load model
    with open(os.path.join(path, 'config.yaml'), 'r') as f: 
        train_config = OmegaConf.create(yaml.safe_load(f))
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'
        model = load_checkpoint(train_config, os.path.join(path, 'models/best.ckpt'), strict=False, map_location='cpu') #FIXME
        model.freeze()
        model.to(device)
        return model


@st.cache(allow_output_mutation=True)
def seg_model():
    model = create_model("Unet_2020-07-20")
    model.eval()
    return model

model = load_model()
people_seg_model = seg_model()

MAX_SIZE = 1024 # 이외 사이즈 실험
transform = albu.Compose(
    [albu.LongestMaxSize(max_size=MAX_SIZE), albu.Normalize(p=1)], p=1
)

option = st.selectbox(
     'How would you like to edit?',
     ('Free Mode','People Segmenation' ))

st.write('You selected:', option)
if option == 'Free Mode':
    # Specify canvas parameters in application
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 50, 25) #edit. default 두께 수정
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"]) 
    drawing_mode = st.sidebar.selectbox("Drawing tool:", ("freedraw", "rect")) #line, circle, transform 제외
    realtime_update = st.sidebar.checkbox("Update in realtime", True)


    HEIGHT_SETTING = 350 # 캔버스 잘림 현상 방지를 위해 캔버스 height 사이즈 고정
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width, # drawing 두께
        stroke_color=stroke_color, # drawing 색
        background_color="#eee", # 캔버스 바탕 색
        background_image=Image.open(bg_image) if bg_image else None, #Pillow image to display behind canvas
        update_streamlit=realtime_update,
        height=HEIGHT_SETTING, # edit. 
        width=int(Image.open(bg_image).size[0]/Image.open(bg_image).size[1]*HEIGHT_SETTING) if bg_image else 600,
        drawing_mode=drawing_mode,
    )

    # Do something interesting with the image data and paths
    if canvas_result.json_data is not None and bg_image is not None: 
        image = Image.open(bg_image) # 원본 이미지
        height = HEIGHT_SETTING
        width = int(image.size[0]/image.size[1]*HEIGHT_SETTING)
        mask = np.zeros((height,width), np.uint8) # 캔버스에 맞춰 마스크 사이즈 조정

        for ob in canvas_result.json_data["objects"]:
            if ob['type'] == 'rect':
                x1,y1,x2,y2 = ob['left'], ob['top'], ob['left'] + ob['width'], ob['top'] + ob['height']
                mask = cv2.rectangle(mask, (x1, y1), (x2, y2), (1), cv2.FILLED)

            if ob['type'] == 'path':
                for dot in ob['path']:
                    if dot[0] != 'Q':
                        continue
                    x1, y1, x2, y2 = map(int, dot[1:])
                    mask = cv2.line(mask, (x1, y1), (x2, y2), (1), stroke_width)
        mask = cv2.resize(mask, (image.size[0], image.size[1])) # (h,w) 마스크를 모델이 입력시 원본 이미지 비율대로 변환

        # inference
        output_image = inference(image, mask, model, mask_dilation=False)
        button(output_image, path)

else:
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        original_image = Image.open(uploaded_file)
        original_image_array = np.array(original_image) #(h, w, c)
        st.image(original_image_array, caption="Before", use_column_width=True)
        st.write("")
        st.write("Detecting people...")

        original_height, original_width = original_image_array.shape[:2]
        image = transform(image=original_image_array)["image"]
        padded_image, pads = pad(image, factor=MAX_SIZE, border=cv2.BORDER_CONSTANT) # image size가 factor로 divisible하도록, pads =(x_min_pad, y_min_pad, x_max_pad, y_max_pad)

        x = torch.unsqueeze(tensor_from_rgb_image(padded_image), 0) # (1, c, h, w)

        with torch.no_grad():
            prediction = people_seg_model(x)[0][0]

        mask = (prediction > 0).cpu().numpy().astype(np.uint8)
        mask = unpad(mask, pads) #Crops patch from the center so that sides are equal to pads.
        mask = cv2.resize(
            mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST
        ) 
        mask_3_channels = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        dst = cv2.addWeighted(
            original_image_array, 1, (mask_3_channels * (0, 255, 0)).astype(np.uint8), 0.5, 0
        ) #imgA * a + imgB * b + c , a+b=1이면 알파블렌딩

        # split result
        # mask = np.uint8((mask > 127) * 255)
        cnt, labels = cv2.connectedComponents(mask)
        h,w = mask.shape
        result = np.zeros((h,w,3), dtype=np.uint8)
        for i in range(1, cnt):
            result[labels==i] = [int(j) for j in np.random.randint(0,255, 3)]

        tmp1 = st.image(mask * 255, caption="Mask", use_column_width=True)
        tmp2 = st.image(dst, caption="Image + mask", use_column_width=True)
        tmp3 = st.image(result, caption="Split Mask", use_column_width=True)

        # select region
        selected_region = st.radio("Select Num", range(1, cnt+1))
        selected_result = np.zeros((h,w,3), dtype=np.uint8)
        selected_result[labels==selected_region] =  [int(j) for j in np.random.randint(0,255, 3)]
        st.image(selected_result, caption="Split Mask", use_column_width=True)
        if st.button('Erase'):
            tmp1, tmp2, tmp3 = st.empty(),st.empty(),st.empty()
            selected_result[labels==selected_region] = [1, 1, 1]
            selected_result = np.transpose(selected_result, (2, 0, 1))[0]
            # inference
            
            output_image = inference(original_image, selected_result, model, mask_dilation=True)

            button(output_image, path)
