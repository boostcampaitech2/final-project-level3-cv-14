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
import copy
sys.path.append(os.path.join(os.getcwd(), 'Utils'))
ImageEncoder = __import__("ImageEncoder")

st.set_page_config(layout="wide")
image_uploaded = st.sidebar.file_uploader("Image Upload:", type=["png", "jpg"])
choice = st.sidebar.selectbox('이미지 보정 도구 선택', ['Inpainting', 'Super Resolution', 'Deblur'])
st.subheader(choice)

if image_uploaded:
    image = Image.open(image_uploaded)
else:
    image = Image.open('WebServer/demo.jpg')



if choice == 'Inpainting':
    # Specify canvas parameters in application
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 50, 35)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    drawing_mode = st.sidebar.selectbox("Drawing tool:", ["freedraw", "line", "rect", "circle", "transform"])

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
    )

    # TODO: 이미지 DB로 전송

    if canvas_result.json_data is not None:
        #print(canvas_result.json_data["objects"])
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

        response = requests.post('http://jiseong.iptime.org:8786/inference/', files={'image': image_bytes, 'mask': mask_bytes})
        image_inpaint = np.fromstring(response.content, np.uint8)
        result = cv2.imdecode(image_inpaint, cv2.IMREAD_COLOR)
        st.image(result, use_column_width=True)

elif choice == 'Super Resolution':
    ratio_list = ['2', '3', '4', '8']
    ratio = st.sidebar.selectbox('이미지 확대 비율 선택', ratio_list)

    st.write("")

    w = st.sidebar.slider("width", 0, image.width, (0, image.width))
    h = st.sidebar.slider("height", 0, image.height, (0, image.height))

    bg = np.zeros_like(image)
    bg[h[0]:h[1], w[0]:w[1]] = 255
    bg = Image.fromarray(bg)
    blended = Image.blend(image, bg, alpha=0.3)
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
            response = requests.post('http://jiseong.iptime.org:8890/super', files=files)  # TODO: change into server addr
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

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=1,  # drawing 두께
        background_color="#eee",  # 캔버스 바탕 색
        background_image=image,  # Pillow image to display behind canvas
        update_streamlit=True,
        height=int(image.size[1]),
        width=int(image.size[0]),
        drawing_mode="rect",
    )

    # TODO: test
    if st.sidebar.button('결과 보기'):
        with st.spinner('Processing...'):
            image = np.array(image.convert('RGB'))
            img_byte_arr = ImageEncoder.Encode(image, ext='jpg', quality=90)
            files = {'image': img_byte_arr}
            response = requests.post('http://jiseong.iptime.org:8891/deblur',
                                     files=files)  # TODO: change into server addr
        if response.status_code == 200:
            new_image = ImageEncoder.Decode(response.content)
            st.success('Done!')
            st.image(new_image, caption='Processed Image', use_column_width=True)
        else:
            st.error('Error Status Code:{}'.format(response.status_code))

    if canvas_result.json_data is not None:  # edit
        height = image.size[1]
        width = image.size[0]
        image = np.array(image)
        img = copy.deepcopy(image)
        mask = np.zeros_like(image)

        for ob in canvas_result.json_data["objects"]:
            x1, y1, x2, y2 = ob['left'], ob['top'], ob['left'] + ob['width'], ob['top'] + ob['height']
            cv2.rectangle(img, (x1, y1), (x2, y2), (0), -1)
            cv2.rectangle(mask, (x1, y1), (x2, y2), (0), -1)
            masked = cv2.bitwise_xor(image, img)

        if st.sidebar.button('선택 영역 결과 보기'):
            with st.spinner('Processing...'):
                masked_byte_arr = ImageEncoder.Encode(masked, ext='jpg', quality=90)
                files = {'image': masked_byte_arr}
                response = requests.post('http://127.0.0.1:8000/deblur',
                                         files=files)  # TODO: change into server addr
            if response.status_code == 200:
                response_arr = ImageEncoder.Decode(response.content)
                transformed_masked = np.where(masked, response_arr, 0)
                new_image = img + transformed_masked
                st.success('Done!')
                st.image(new_image, caption='Processed Image', use_column_width=True)
            else:
                st.error('Error Status Code:{}'.format(response.status_code))