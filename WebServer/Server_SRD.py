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
import sys
sys.path.append(os.getcwd())
from Utils import ImageEncoder
from utils import load_img

def main():
  st.set_page_config(layout="wide")
  #st.title("종합 이미지 보정 도구 ")
  st.sidebar.header('Choose Options')

  #module 선택
  menu = ['Image Completion','Super Resolution','Deblur']
  choice = st.sidebar.selectbox('이미지 보정 도구 선택', menu)
  if choice:
    st.subheader(choice)

  #파일 선택
  uploaded_file = st.sidebar.file_uploader("Choose an image...", type="jpg",key="SR")
  
  #모듈 실행
  ### Image Completion ###
  if choice=='Image Completion':
    pass
    #TODO: Image Completion 모듈 추가
  
  ### Super Resolution ###
  elif choice=='Super Resolution':
    ratio_list = ['2','3','4','8']
    ratio = st.sidebar.selectbox('이미지 확대 비율 선택',ratio_list)

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
          response = requests.post('http://127.0.0.1:8000/super',files=files) #TODO: change into server addr
        if response.status_code==200:
          new_image = ImageEncoder.Decode(response.content)
          st.success('Done!')
          col1, col2 = st.columns(2)
          col1.image(cropped_img, caption='Cropped Image', use_column_width=True)
          col2.image(new_image, caption='Processed Image', use_column_width=True)
        else:
          st.error('Error Status Code:{}'.format(response.status_code))

  ### Deblur ###
  elif choice=='Deblur':
    if uploaded_file is not None:
      image = load_img(uploaded_file)
      st.image(image, caption='Original Image', use_column_width=True)
      st.write("")

      #TODO: image crop
      if st.sidebar.button('결과 보기'):
        with st.spinner('Processing...'):
          files = {'files':uploaded_file.getvalue()}
          response = requests.post('http://127.0.0.1:8000/deblur',files=files) #TODO: change into server addr
        if response.status_code==200:
          bytes_data = io.BytesIO(response.content)
          new_image = Image.open(bytes_data)
          st.success('Done!')
          st.image(new_image, caption='Processed Image', use_column_width=True)
        else:
          st.error('Error Status Code:{}'.format(response.status_code))

if __name__ == '__main__':
	main()