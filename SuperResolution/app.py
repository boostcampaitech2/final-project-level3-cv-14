import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from SwinIR.models.network_swinir import SwinIR as net
import torch
import cv2
import argparse
import os
import tensorflow as tf
import SRNDeblur.models.model_streamlit as srn_model

BASE_DIR = './drive/MyDrive/streamlit/'
MODEL_PATH = BASE_DIR + 'experiments/pretrained_models/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth'

#--task real_sr --scale 4 --model_path model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth --folder_lq testsets/RealSRSet+5images --tile
def define_model():
  # use 'nearest+conv' to avoid block artifacts
  model = net(upscale=4, in_chans=3, img_size=64, window_size=8,
              img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
              mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')

  param_key_g = 'params_ema'
  pretrained_model = torch.load(MODEL_PATH)
  model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
  return model

def predict_SR(image):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = define_model()
  model.eval()
  model = model.to(device)
  
  border = 0
  window_size = 8
  img_gt = None
  img_array = np.array(image)
  #st.write(img_array)
  img_lq = cv2.cvtColor(img_array, cv2.IMREAD_COLOR).astype(np.float32) / 255.

  # read image
  img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
  img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

  # inference
  with torch.no_grad():
      # pad input image to be a multiple of window_size
      _, _, h_old, w_old = img_lq.size()
      h_pad = (h_old // window_size + 1) * window_size - h_old
      w_pad = (w_old // window_size + 1) * window_size - w_old
      img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
      img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
      output = model(img_lq)
      output = output[..., :h_old * 4, :w_old * 4]

  # save image
  output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
  if output.ndim == 3:
      output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
  output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
  #cv2.imwrite(f'{save_dir}/{imgname}_SwinIR.png', output)

  return output

def parse_args():
    parser = argparse.ArgumentParser(description='deblur arguments')
    parser.add_argument('--phase', type=str, default='test', help='determine whether train or test')
    parser.add_argument('--datalist', type=str, default=BASE_DIR+'SRNDeblur/datalist_gopro.txt', help='training datalist')
    parser.add_argument('--model', type=str, default='color', help='model type: [lstm | gray | color]')
    parser.add_argument('--batch_size', help='training batch size', type=int, default=16)
    parser.add_argument('--epoch', help='training epoch number', type=int, default=4000)
    parser.add_argument('--lr', type=float, default=1e-4, dest='learning_rate', help='initial learning rate')
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0', help='use gpu or cpu')
    parser.add_argument('--height', type=int, default=720,
                        help='height for the tensorflow placeholder, should be multiples of 16')
    parser.add_argument('--width', type=int, default=1280,
                        help='width for the tensorflow placeholder, should be multiple of 16 for 3 scales')
    parser.add_argument('--input_path', type=str, default=BASE_DIR+'SRN-Deblur/testing_set',
                        help='input path for testing images')
    parser.add_argument('--output_path', type=str, default=BASE_DIR+'SRN-Deblur/testing_res',
                        help='output path for testing images')
    args = parser.parse_args()
    return args

def predict_DB(image):
  args = parse_args()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  deblur = srn_model.DEBLUR(args)
  output = deblur.test(args.height, args.width, np.array(image))
  return output


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

  uploaded_file = st.file_uploader("Choose an image...", type="jpg",key="SR")
  if uploaded_file is not None:
      image = Image.open(uploaded_file)
      st.image(image, caption='Original Image', use_column_width=False)
      st.write("")
      new_image = predict_SR(image)
      st.image(new_image, caption='Processed Image', use_column_width=True)

  ### Deblur ###
  st.title("Deblur")
  st.write("Blurry 이미지를 보정해줍니다.")

  uploaded_file = st.file_uploader("Choose an image...", type="jpg",key="DB")
  if uploaded_file is not None:
      image = Image.open(uploaded_file)
      st.image(image, caption='Original Image', use_column_width=False)
      st.write("")
      new_image = predict_DB(image)
      st.image(new_image, caption='Processed Image', use_column_width=True)

if __name__ == '__main__':
	main()