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
	#st.title("Awesome Streamlit for MLDDD")
	#st.subheader("How to run streamlit from colab")
  st.write("""
  # 종합 이미지 보정 도구

  Image completion, Super resolution, Deblur
  
  """)

  ### Super Resolution ###
  st.title("Super Resolution")
  st.write("이미지의 해상도를 높여줍니다.")

  uploaded_file1 = st.file_uploader("Choose an image...", type="jpg",key="SR")
  if uploaded_file1 is not None:
      image = Image.open(uploaded_file1)
      st.image(image, caption='Original Image', use_column_width=False)
      st.write("")
      #TODO: image crop, resize scale 선택
      #TODO: sidebar에서 선택한 args 넘겨주기

      ###기존 함수 추론 start ###
      #new_image = sr_predictor().predict(image) 
      ###기존 함수 추론 end ###
      
      ### Fast API 추론 start ###
      files = {'files':uploaded_file1.getvalue()}
      response = requests.post('http://127.0.0.1:8000/super',files=files) #TODO: change into server addr
      bytes_data = io.BytesIO(response.content)
      new_image = Image.open(bytes_data)
      ### Fast API 추론 end ###

      st.image(new_image, caption='Processed Image', use_column_width=True)

  ### Deblur ###
  st.title("Deblur")
  st.write("Blurry 이미지를 보정해줍니다.")

  uploaded_file2 = st.file_uploader("Choose an image...", type="jpg",key="DB")
  if uploaded_file2 is not None:
      image = Image.open(uploaded_file2)
      st.image(image, caption='Original Image', use_column_width=False)
      st.write("")
      #TODO: image crop

      ###기존 함수 추론 start ###
      #new_image = db_predictor().predict(image)
      ###기존 함수 추론 end ###
      
      ### Fast API 추론 start ###
      files = {'files':uploaded_file2.getvalue()}
      response = requests.post('http://127.0.0.1:8000/deblur',files=files) #TODO: change into server addr
      bytes_data = io.BytesIO(response.content)
      new_image = Image.open(bytes_data)
      ### Fast API 추론 end ###

      #TODO: image combine
      st.image(new_image, caption='Processed Image', use_column_width=True)

if __name__ == '__main__':
	main()