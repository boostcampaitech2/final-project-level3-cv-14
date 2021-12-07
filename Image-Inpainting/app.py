import sys
import os
module_path = os.path.join(os.getcwd(), 'lama') 
sys.path.append(module_path)
import cv2
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate


from lama.saicinpainting.training.trainers import load_checkpoint
from lama.saicinpainting.evaluation.data import pad_img_to_modulo
from lama.saicinpainting.evaluation.utils import move_to_device

import streamlit as st
from streamlit_drawable_canvas import st_canvas


import PIL.Image as Image
import time
from io import BytesIO #edit
import base64 #edit
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
    
def button(output_img, save_dir):
        """ 아웃풋 이미지를 다운로드받는 버튼 생성
        
        Keyword arguments:
        output_img : PIL image
        save_dir : 이미지를 저장할 디렉토리

        """
        IMAGE_SAVE_PATH = os.path.join(save_dir, 'result.jpg')
        output_img.save(IMAGE_SAVE_PATH, format="JPEG")
        with open(IMAGE_SAVE_PATH, "rb") as fp:
            btn = st.download_button(
                label="Download Your Image",
                data=fp,
                file_name='result.jpg',
                mime="image/jpeg",
            )
            
            
model = load_model()


# Specify canvas parameters in application
stroke_width = st.sidebar.slider("Stroke width: ", 1, 50, 25) #edit. default 두께 수정
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"]) 
drawing_mode = st.sidebar.selectbox("Drawing tool:", ("freedraw", "rect")) #edit. line, circle, transform 제외
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
if canvas_result.json_data is not None and bg_image is not None: #edit
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
    mask = cv2.resize(mask, (image.size[0], image.size[1])) # edit. (h,w) 마스크를 모델이 입력시 원본 이미지 비율대로 변환

    # inference
    img = np.array(image.convert('RGB'))
    img = np.transpose(img, (2, 0, 1))
    img = img.astype('float32') / 255

    result = dict(image=img, mask=mask[None, ...])
    result['image'] = pad_img_to_modulo(result['image'], 8)
    result['mask'] = pad_img_to_modulo(result['mask'], 8)

    batch = move_to_device(default_collate([result]), device)

    batch['mask'] = (batch['mask'] > 0) * 1 # 0을 넘는 경우는 True이고, True를 1을 곱해 정수로 만들어 0과 1만 가지도록 만듦 (resize 할때 보간법 등이 사용되기 때문에 0~1 사이 값이 생기는 경우가 있어 이런 처리가 필요)
    batch = model(batch)
    cur_res = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()  
    st.image(cur_res, use_column_width='auto', caption="After")
        
    pillow_image = Image.fromarray(np.clip(cur_res * 255, 0, 255).astype('uint8')) # 0~1 범위 -> 0~255범위로 조정
        
    button(pillow_image, path)
