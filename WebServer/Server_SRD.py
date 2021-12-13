import numpy as np
import pandas as pd
from PIL import Image
import PIL
import cv2
import torch
import tensorflow as tf
import os
import argparse
import streamlit as st
import io
import base64
import requests

from streamlit_drawable_canvas import st_canvas

import sys
import copy

sys.path.append(os.getcwd())
from Utils import ImageEncoder
from WebServer.utils import load_img


def main():
    st.set_page_config(layout="wide")
    st.sidebar.header('Choose Options')

    # module 선택
    menu = ['Image Completion', 'Super Resolution', 'Deblur']
    choice = st.sidebar.selectbox('이미지 보정 도구 선택', menu)
    if choice:
        st.subheader(choice)

    # 파일 선택
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"], key="SR")

    # 모듈 실행
    ### Image Completion ###
    if choice == 'Image Completion':
        pass
        # TODO: Image Completion 모듈 추가

    ### Super Resolution ###
    elif choice == 'Super Resolution':
        ratio_list = ['2', '3', '4', '8']
        ratio = st.sidebar.selectbox('이미지 확대 비율 선택', ratio_list)

        if uploaded_file is not None:
            image = load_img(uploaded_file)
            st.write("")

            w = st.sidebar.slider("width", 0, image.width, (0, image.width))
            h = st.sidebar.slider("height", 0, image.height, (0, image.height))

            bg = np.zeros_like(image)
            bg[h[0]:h[1], w[0]:w[1]] = 255
            bg = Image.fromarray(bg)
            blended = Image.blend(image, bg, alpha = 0.3)
            st.image(blended, caption='Original Image', use_column_width=True)

            if st.sidebar.button('결과 보기'):
                with st.spinner('Processing...'):
                    cropped_img = image.crop((w[0], h[0], w[1], h[1]))
                    image = np.array(cropped_img.convert('RGB'))
                    img_byte_arr = ImageEncoder.Encode(image, ext='jpg', quality=90)
                    files = {
                        'image': img_byte_arr,
                        'ratio': (None, int(ratio)),
                    }
                    response = requests.post('http://127.0.0.1:8000/super',
                                             files=files)  # TODO: change into server addr
                if response.status_code == 200:
                    new_image = ImageEncoder.Decode(response.content)
                    st.success('Done!')
                    col1, col2 = st.columns(2)
                    col1.image(cropped_img, caption='Cropped Image', use_column_width=True)
                    col2.image(new_image, caption='Processed Image', use_column_width=True)
                else:
                    st.error('Error Status Code:{}'.format(response.status_code))

    ### Deblur ###
    elif choice == 'Deblur':
        if uploaded_file is not None:
            image = load_img(uploaded_file)
            st.write("")
            
            canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=1, # drawing 두께
            background_color="#eee", # 캔버스 바탕 색
            background_image=Image.open(uploaded_file) if uploaded_file else None, #Pillow image to display behind canvas
            update_streamlit=True,
            height = int(Image.open(uploaded_file).size[1]),
            width = int(Image.open(uploaded_file).size[0]),
            drawing_mode="rect",
            )
            
            if canvas_result.json_data is not None and uploaded_file is not None: #edit
                height = image.size[1]
                width = image.size[0]
                image = np.array(image)
                img = copy.deepcopy(image)
                mask = np.zeros_like(image)

                for ob in canvas_result.json_data["objects"]:
                    x1,y1,x2,y2 = ob['left'], ob['top'], ob['left'] + ob['width'], ob['top'] + ob['height']
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0), -1)
                    cv2.rectangle(mask, (x1, y1), (x2, y2), (0), -1)   

                    masked = cv2.bitwise_xor(image, img)

            # TODO: test
            if st.sidebar.button('결과 보기'):
                with st.spinner('Processing...'):
                    files = {'files': uploaded_file.getvalue()}
                    response = requests.post('http://127.0.0.1:8000/deblur',
                                             files=files)  # TODO: change into server addr
                if response.status_code == 200:
                    bytes_data = io.BytesIO(response.content)
                    new_image = Image.open(bytes_data)
                    st.success('Done!')
                    st.image(new_image, caption='Processed Image', use_column_width=True)
                else:
                    st.error('Error Status Code:{}'.format(response.status_code))
                    
            elif st.sidebar.button('선택 영역 결과 보기'):
                with st.spinner('Processing...'):
                        files = {'files': masked}
                        response = requests.post('http://127.0.0.1:8000/deblur',
                                                 files=files)  # TODO: change into server addr
                if response.status_code == 200:
                    transformed_masked = np.where(masked, response, 0)
                    transformed_image = img + transformed_masked
                    bytes_data = io.BytesIO(transformed_image.content)
                    new_image = Image.open(bytes_data)
                    st.success('Done!')
                    st.image(new_image, caption='Processed Image', use_column_width=True)
                else:
                    st.error('Error Status Code:{}'.format(response.status_code))

if __name__ == '__main__':
    main()
