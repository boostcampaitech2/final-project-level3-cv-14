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
from SuperRes.predict import Predictor as sr_predictor
from Deblur.predict import Predictor as db_predictor

def main():
  st.title("종합 이미지 보정 도구 ")
  st.sidebar.header('Choose Options')

  #module 선택
  menu = ['Image Completion','Super Resolution','Deblur']
  choice = st.sidebar.selectbox('이미지 보정 도구 선택', menu)
  if choice:
    st.subheader(choice)

  #파일 선택
  uploaded_file = st.sidebar.file_uploader("Choose an image...", type="jpg",key="SR")
  if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Original Image', use_column_width=True)
    st.write("")
  
  #모듈 실행
  ### Image Completion ###
  if choice=='Image Completion':
    pass
    #TODO: Image Completion 모듈 추가
  
  ### Super Resolution ###
  elif choice=='Super Resolution':
    ratio_list = ['2','3','4','8']
    ratio = st.sidebar.selectbox('이미지 확대 비율 선택',ratio_list)
    #TODO: image crop, resize scale 선택
    #TODO: sidebar에서 선택한 args 넘겨주기
    if st.sidebar.button('결과 보기'):
      with st.spinner('Processing...'):
        files = {'files':uploaded_file.getvalue()}
        response = requests.post('http://127.0.0.1:8000/super',files=files) #TODO: change into server addr
        bytes_data = io.BytesIO(response.content)
        new_image = Image.open(bytes_data)
      st.success('Done!')
      st.image(new_image, caption='Processed Image', use_column_width=True)
  
  ### Deblur ###
  elif choice=='Deblur':
    #TODO: image crop
    if st.sidebar.button('결과 보기'):
      with st.spinner('Processing...'):
        files = {'files':uploaded_file.getvalue()}
        response = requests.post('http://127.0.0.1:8000/deblur',files=files) #TODO: change into server addr
        bytes_data = io.BytesIO(response.content)
        new_image = Image.open(bytes_data)
      st.success('Done!')
      st.image(new_image, caption='Processed Image', use_column_width=True)

if __name__ == '__main__':
	main()