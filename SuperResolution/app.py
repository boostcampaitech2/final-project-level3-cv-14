import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import torch
import cv2
import argparse
import os
import tensorflow as tf
from SuperRes.predict import Predictor

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
      sr_predictor = Predictor()
      new_image = sr_predictor.predict(image) #TODO: sidebar에서 선택한 args 넘겨주기
      st.image(new_image, caption='Processed Image', use_column_width=True)
      #st.write("(psnr, ssim, psnr_y, ssim_y) = ",score)

#   ### Deblur ###
#   st.title("Deblur")
#   st.write("Blurry 이미지를 보정해줍니다.")

#   uploaded_file2 = st.file_uploader("Choose an image...", type="jpg",key="DB")
#   if uploaded_file2 is not None:
#       image = Image.open(uploaded_file2)
#       st.image(image, caption='Original Image', use_column_width=False)
#       st.write("")
#       new_image = predict_DB(image)
#       st.image(new_image, caption='Processed Image', use_column_width=True)

if __name__ == '__main__':
	main()