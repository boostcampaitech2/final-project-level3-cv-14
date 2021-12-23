'''
이 모듈은 다음과 같은 기능을 제공합니다.
 - Streamlit 기반의 이미지 편집 웹서버를 구동합니다.
 - 세 가지 AI 이미지 복원 기능을 RestAPI 방식으로 호출합니다.
  - Inpainting : 이미지에서 특정 영역을 지우고, 지운 자리에 자연스러운 이미지를 추론해서 합성하는 기능을 제공합니다.
  - Super Resolution : 추론을 통한 자연스러운 업샘플링 기능을 제공합니다.
  - Deblur : 추론을 통해 자연스럽게 모션블러를 제거합니다.
 - 잘라내기, 배율을 통해 원하는 부분을 축소, 확대할 수 있습니다.
 - "원본", "뒤로", "앞으로" 히스토리 도구 : 세가지 히스토리 기능을 통해 과거 작업을 잃어버리지 않도록 편의 기능을 제공합니다.
 - "그리기 도구" : Free Draw, Rect 기능으로 원하는 영역을 선택할 수 있는 기능을 제공합니다.
 - 입력 이미지 정보, 추론 이미지 정보, 사용자 스코어는 Cloud SQL DB의 각각의 테이블에 저장되며 이미지 byte는 Cloud Storage에 저장됩니다. 

작성자 김지성, 홍지연
최종 수정일 2021-12-16
'''

import sys, os
import requests
import time

import numpy as np
import cv2
import PIL.Image as Image

import streamlit as st
from streamlit_drawable_canvas import st_canvas


import uuid
from utils import send_to_bucket
from db import insert_data_input, insert_data_inference, insert_data_score 


sys.path.append(os.path.join(os.getcwd(), 'Utils'))
ImageEncoder = __import__("ImageEncoder")
Storage = __import__("Storage")
DB = __import__("DB")
from ErrorChecker import Sentry

st.set_page_config(layout="wide")


def RefreshCanvas():
    st.session_state["canvas_id"] = time.time()
    st.session_state["mask"] = None

if st.session_state.get("canvas_id") is None:
    st.session_state["canvas_id"] = time.time()

if st.session_state.get("magnification") is None:
    st.session_state["magnification"] = 1

if st.session_state.get("history") is None:
    st.session_state["history"] = []

if st.session_state.get("history_idx") is None:
    st.session_state["history_idx"] = 0
    
if st.session_state.get('is_image_in_input_table') is None:
    st.session_state['is_image_in_input_table'] = False
    
if st.session_state.get('input_id') is None:
    st.session_state['input_id'] = uuid.uuid4().hex
    
if st.session_state.get('inference_index') is None:
    st.session_state['inference_index'] = 0

    
def insert_input_table(input_id, image_bytes):
    if st.session_state['is_image_in_input_table']==False:
        input_url = Storage.send_to_bucket(input_id, image_bytes)
        DB.insert_data_input(input_id, input_url)
        st.session_state['is_image_in_input_table']=True
    else:
        pass

    
def insert_inference_table(input_id, inference_type, output_image_bytes):
    inference_index = st.session_state['inference_index']
    inference_url = Storage.send_to_bucket(input_id+f'_{inference_index}', output_image_bytes)
    DB.insert_data_inference(input_id, inference_url, inference_type)
    st.session_state['inference_index'] += 1


def main():
    # 만약 이미지를 업로드 했다면 원본 이미지를 업로드이미지로 설정, 아니라면 데모 이미지로 설정
    image_uploaded = st.sidebar.file_uploader("Image Upload:", type=["png", "jpg"])
    if image_uploaded:
        image_origin = Image.open(image_uploaded)
    else:
        image_origin = Image.open('WebServer/demo.jpg')
    image_origin = np.array(image_origin.convert('RGB'))

    # 새 이미지를 업로드 했다면 image_current를 업데이트
    flag_newImage = st.session_state.get("image_origin") is None or not np.array_equal(st.session_state["image_origin"], image_origin)
    if flag_newImage:
        # 새로 업로드
        st.session_state["image_origin"] = image_origin
        st.session_state["image_current"] = image_origin
        st.session_state['is_image_in_input_table'] = False
        st.session_state['input_id'] = uuid.uuid4().hex
        st.session_state['inference_index'] = 0
        RefreshCanvas()

    st.sidebar.text("AI 복원")
    flag_inpainting = st.sidebar.button('Inpainting')
    flag_superResolution = st.sidebar.button('Super Resolution')
    flag_deblur = st.sidebar.button('Deblurring')

    st.sidebar.text("이미지 편집")
    edit_col1, edit_col2 = st.sidebar.columns(2)
    flag_crop = edit_col1.button('잘라내기')

    magnification = st.sidebar.slider("배율 ", 0.1, 5., 1.)
    if st.session_state["magnification"] != magnification:
        st.session_state["magnification"] = magnification
        RefreshCanvas()

    st.sidebar.text("히스토리")
    history_col1, history_col2, history_col3 = st.sidebar.columns(3)
    flag_history_origin = history_col1.button("원본")
    flag_history_back = history_col2.button("뒤로")
    if len(st.session_state["history"])-1 > st.session_state["history_idx"]:
        flag_history_front = history_col3.button("앞으로")
    else:
        flag_history_front = False

    # 원본 이미지 출력
    st.sidebar.text(f"원본 이미지 {st.session_state['image_origin'].shape[:2]}")
    st.sidebar.image(st.session_state["image_origin"])

    st.sidebar.text(f"현재 이미지 {st.session_state['image_current'].shape[:2]}")
    st.sidebar.image(st.session_state["image_current"])

    if flag_inpainting:
        if st.session_state.get("mask") is not None:
            image_bytes = ImageEncoder.Encode(st.session_state["image_current"], ext='jpg', quality=90)
            mask_bytes = ImageEncoder.Encode(st.session_state["mask"], ext='png')
            response = requests.post('http://jiseong.iptime.org:8786/inference/', files={'image': image_bytes, 'mask': mask_bytes})
            st.session_state["image_current"] = ImageEncoder.Decode(response.content)
            
            insert_input_table(st.session_state['input_id'], image_bytes)
            insert_inference_table(input_id=st.session_state['input_id'], inference_type='inpainting', output_image_bytes=response.content)

            RefreshCanvas()
        else:
            st.error("Inpainting을 위해 영역을 선택해주세요!")

    elif flag_superResolution:
        if st.session_state.get("mask") is None:
            st.session_state["mask"] = np.ones(st.session_state["image_current"].shape[:2], dtype=np.uint8)
        mask_front = st.session_state["mask"]
        mask_background = np.array(st.session_state["mask"]==0,dtype=np.uint8)

        image_bytes = ImageEncoder.Encode(st.session_state["image_current"], ext='jpg', quality=90)
        response = requests.post('http://jiseong.iptime.org:8788/super', files={'image': image_bytes})
        result = ImageEncoder.Decode(response.content)

        image_quarter = cv2.resize(st.session_state["image_current"], dsize=(0,0), fx=4, fy=4)
        mask_front_quarter = cv2.resize(mask_front, dsize=(0,0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        mask_background_quarter = cv2.resize(mask_background, dsize=(0, 0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST)

        image_front = cv2.bitwise_and(result, result, mask=mask_front_quarter)
        image_background = cv2.bitwise_and(image_quarter, image_quarter, mask=mask_background_quarter)

        st.session_state["image_current"] = image_front+image_background

        insert_input_table(st.session_state['input_id'], image_bytes)
        insert_inference_table(input_id=st.session_state['input_id'], inference_type='superResolution', output_image_bytes=response.content)
        
        RefreshCanvas()

    elif flag_deblur:
        if st.session_state.get("mask") is None:
            st.session_state["mask"] = np.ones(st.session_state["image_current"].shape[:2], dtype=np.uint8)
        mask_front = st.session_state["mask"]
        mask_background = np.array(st.session_state["mask"]==0,dtype=np.uint8)

        image_bytes = ImageEncoder.Encode(st.session_state["image_current"], ext='jpg', quality=90)
        response = requests.post('http://jiseong.iptime.org:8789/deblur', files={'image': image_bytes})  # TODO: change into server addr
        result = ImageEncoder.Decode(response.content)

        image_front = cv2.bitwise_and(result, result, mask=mask_front)
        image_background = cv2.bitwise_and(st.session_state["image_current"],st.session_state["image_current"],mask=mask_background)

        st.session_state["image_current"] = image_front+image_background
        insert_input_table(st.session_state['input_id'], image_bytes)
        insert_inference_table(input_id=st.session_state['input_id'], inference_type='deblur', output_image_bytes=response.content)
        
        RefreshCanvas()

    elif flag_crop:
        y, x = np.nonzero(st.session_state["mask"])
        x1, y1, x2, y2 = np.min(x), np.min(y), np.max(x), np.max(y)
        st.session_state["image_current"] = st.session_state["image_current"][y1:y2, x1:x2]
        
        RefreshCanvas()

    elif flag_history_back:
        if st.session_state["history_idx"] > 0:
            st.session_state["history_idx"] -= 1
        st.session_state["image_current"] = st.session_state["history"][st.session_state["history_idx"]]
        RefreshCanvas()

    elif flag_history_front:
        if len(st.session_state["history"]) > st.session_state["history_idx"]:
            st.session_state["history_idx"] += 1
        st.session_state["image_current"] = st.session_state["history"][st.session_state["history_idx"]]
        RefreshCanvas()

    elif flag_history_origin:
        st.session_state["image_current"] = st.session_state["image_origin"]
        st.session_state["history_idx"] = 0

    if flag_newImage:
        st.session_state["history"] = [st.session_state["image_current"]]
        st.session_state["history_idx"] = 0

    if any([flag_inpainting, flag_superResolution, flag_deblur, flag_crop]):
        st.session_state["history"].append(st.session_state["image_current"])
        st.session_state["history_idx"] = len(st.session_state["history"]) - 1

    # 영역 선택 도구
    drawing_mode = st.selectbox("영역 선택 도구:", ["Rect", "Free Draw", "Inpainting 영역 추천"])
    if drawing_mode == "Free Draw":
        tool = "freedraw"
        stroke_width = st.slider("Stroke width: ", 1, 50, 35)

    elif drawing_mode == "Rect":
        tool = "rect"
        stroke_width = 1

    elif drawing_mode == "Inpainting 영역 추천":
        tool = "freedraw"
        stroke_width = 1

    if drawing_mode == "Inpainting 영역 추천":
        image_bytes = ImageEncoder.Encode(st.session_state["image_current"], ext='jpg', quality=90)
        response = requests.post('http://jiseong.iptime.org:8790/inference', files={'image': image_bytes})  # TODO: change into server addr
        image_mask = ImageEncoder.Decode(response.content, channels=1)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        image_mask = cv2.dilate(image_mask, k) # 팽창연산
        image_mask = np.array(image_mask > 0.5, dtype=np.uint8)

        cnt, labels = cv2.connectedComponents(image_mask)
        image_mask = cv2.cvtColor(image_mask, cv2.COLOR_GRAY2RGB)

        # 랜덤컬러 segment 방식
        #result = np.zeros(st.session_state["image_current"].shape, dtype=np.uint8)
        #for i in range(1, cnt):
            #result[labels == i] = [int(j) for j in np.random.randint(0, 255, 3)]
        #image_view = cv2.addWeighted(st.session_state["image_current"], 0.5, result.astype(np.uint8), 0.5, 0)

        # 밝기만 조절하는 방식
        image_view = cv2.addWeighted(st.session_state["image_current"], 0.7, (image_mask * (255, 255, 255)).astype(np.uint8), 0.3, 0)
        st.session_state["image_segments_components"] = labels
    else:
        image_view = st.session_state["image_current"]
        st.session_state["image_segments_components"] = None

    # 캔버스에 보여줄 이미지 * 배율
    image_view = cv2.resize(image_view, dsize=(0, 0), fx=magnification, fy=magnification)

    # 캔버스 (이미지 업데이트를 위해 가장 마지막에 위치)
    h,w = image_view.shape[:2]
    drawing_objects = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,  # drawing 두께
        background_color="#eee",  # 캔버스 바탕 색
        background_image=Image.fromarray(image_view),
        update_streamlit=True,
        height=h,
        width=w,
        drawing_mode=tool,
        key=str(st.session_state["canvas_id"])
    )

    # 마스크 생성
    flag_draw = drawing_objects.json_data is not None and drawing_objects.json_data["objects"]  # draw 내용 유무
    if flag_draw:
        if drawing_mode in ["Free Draw", "Rect"]:
            mask = np.zeros((h, w), np.uint8)
            for ob in drawing_objects.json_data["objects"]:
                if ob['type'] == 'rect':
                    x1, y1, x2, y2 = ob['left'], ob['top'], ob['left'] + ob['width'], ob['top'] + ob['height']
                    mask = cv2.rectangle(mask, (x1, y1), (x2, y2), (1), cv2.FILLED)
                if ob['type'] == 'path':
                    for dot in ob['path']:
                        if dot[0] != 'Q':
                            continue
                        x1, y1, x2, y2 = map(int, dot[1:])
                        mask = cv2.line(mask, (x1, y1), (x2, y2), (1), stroke_width)

            h, w = st.session_state["image_current"].shape[:2]
            st.session_state["mask"] = cv2.resize(mask, dsize=(w, h))

        elif drawing_mode == "Inpainting 영역 추천":
            for ob in drawing_objects.json_data["objects"]:
                for dot in ob['path']:
                    if dot[0] != 'L':
                        continue
                    x,y = map(int, dot[1:])
                    st.session_state["image_segments_components_select"] = st.session_state["image_segments_components"][y][x]
                    break

            ma = st.session_state["image_segments_components"] == st.session_state["image_segments_components"][y][x]
            ma = np.array(ma, dtype=np.uint8)
            st.session_state["mask"] = ma

            st.text("선택된 영역")
            st.image(cv2.addWeighted(st.session_state["image_current"], 0.7, (cv2.cvtColor(ma,cv2.COLOR_GRAY2RGB) * (255, 255, 255)).astype(np.uint8), 0.3, 0))




    # 이미지 다운로드
    st.download_button(label="Image Download", data=ImageEncoder.Encode(cv2.cvtColor(st.session_state["image_current"],cv2.COLOR_RGB2BGR)), file_name="image.jpg")
    
    # 별점
    score = st.radio("이 앱을 평가해주세요!",('5점', '4점', '3점', '2점', '1점'))
    image_star = Image.open('WebServer/star.png')
    cols = st.columns(20)
    for idx in range(int(score[0])):
        cols[idx].image(image_star)
    
    if st.button("평가하기"):
        insert_data_score(st.session_state['input_id'], score[0])


if __name__ == "__main__":
    sentry = Sentry()
    try:
        main()
    except Exception as e:
        sentry.check(e)