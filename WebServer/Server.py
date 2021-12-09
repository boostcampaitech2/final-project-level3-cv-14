'''
작성자 김지성, 홍지연
최종 수정일 2021-12-09
'''

import sys, os
import numpy as np
import cv2
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import PIL.Image as Image
import requests
sys.path.append(os.path.join(os.getcwd(), '../Utils'))
ImageEncoder = __import__("ImageEncoder")

# Specify canvas parameters in application
stroke_width = st.sidebar.slider("Stroke width: ", 1, 50, 35)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
drawing_mode = st.sidebar.selectbox("Drawing tool:", ("freedraw", "line", "rect", "circle", "transform"))


if bg_image:
    image = Image.open(bg_image)

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=image,
        update_streamlit=True,
        height=image.size[1],
        width=image.size[0],
        drawing_mode=drawing_mode,
        initial_drawing={},
        key="canvas"
    )

    # TODO: 이미지 DB로 전송

    if canvas_result.json_data is not None:
        print(canvas_result.json_data["objects"])
        image = np.array(image.convert('RGB'))
        mask = np.zeros(image.shape[:2], np.uint8)

        for ob in canvas_result.json_data["objects"]:
            if ob['type'] == 'rect':
                x1, y1, x2, y2 = ob['left'], ob['top'], ob['left'] + ob['width'], ob['top'] + ob['height']
                mask = cv2.rectangle(mask, (x1, y1), (x2, y2), (1), cv2.FILLED)

            if ob['type'] == 'path':
                for dot in ob['path']:
                    if dot[0] != 'Q':
                        continue
                    x1, y1, x2, y2 = map(int, dot[1:])
                    mask = cv2.line(mask, (x1, y1), (x2, y2), (1), stroke_width)

        image_bytes = ImageEncoder.Encode(image, ext='jpg', quality=90)
        mask_bytes = ImageEncoder.Encode(mask, ext='png')

        response = requests.post('http://127.0.0.1:8786/inference/', files={'image': image_bytes, 'mask': mask_bytes})
        image_inpaint = np.fromstring(response.content, np.uint8)
        result = cv2.imdecode(image_inpaint, cv2.IMREAD_COLOR)
        st.image(result)