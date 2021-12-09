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
  menu = ['Image Completion','Super Resolution','Deblur']
  st.sidebar.header('Mode Selection')
  choice = st.sidebar.selectbox('이미지 보정 도구 선택', menu)
  uploaded_file = st.sidebar.file_uploader("Choose an image...", type="jpg",key="SR")
  execute_btn = st.sidebar.button('결과 보기')

  if choice=='Image Completion':
    st.write('image completion')
  else:
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Original Image', use_column_width=False)
        st.write("")

    if execute_btn:
      ### Super Resolution ###
      if choice=='Super Resolution':
        pass
          #TODO: image crop, resize scale 선택
          #TODO: sidebar에서 선택한 args 넘겨주기
          ###기존 함수 추론 start ###
          #new_image = sr_predictor().predict(image) 
          ###기존 함수 추론 end ###

      ### Deblur ###
      elif choice=='Deblur':
        pass
        #TODO: image crop
        ###기존 함수 추론 start ###
        #new_image = db_predictor().predict(image)
        ###기존 함수 추론 end ###

      ### Fast API 추론 start ###
      files = {'files':uploaded_file.getvalue()}
      response = requests.post('http://127.0.0.1:8000/super',files=files) #TODO: change into server addr
      bytes_data = io.BytesIO(response.content)
      new_image = Image.open(bytes_data)
      ### Fast API 추론 end ###

      st.image(new_image, caption='Processed Image', use_column_width=True)

if __name__ == '__main__':
	main()