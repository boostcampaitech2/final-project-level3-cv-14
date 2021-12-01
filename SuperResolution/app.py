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
from SwinIR.utils import util_calculate_psnr_ssim as util

BASE_DIR = './drive/MyDrive/streamlit/'

@st.cache
def define_model(model_kind):
  if model_kind=='real_sr':
    #--task real_sr --scale 4 --model_path model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth --folder_lq testsets/RealSRSet+5images --tile
    MODEL_PATH = BASE_DIR + 'experiments/pretrained_models/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth'
    model = net(upscale=4, in_chans=3, img_size=64, window_size=8,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
    param_key_g = 'params_ema'

  elif model_kind=='lightweight_sr':
    #--task lightweight_sr --scale 4 --model_path model_zoo/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth --folder_lq testsets/Set5/LR_bicubic/X4 --folder_gt testsets/Set5/HR
    MODEL_PATH = BASE_DIR + 'experiments/pretrained_models/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth'
    model = net(upscale=4, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
    param_key_g = 'params'
 
  pretrained_model = torch.load(MODEL_PATH)
  model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
  return model

def predict_SR(image,model_kind='lightweight_sr'):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = define_model(model_kind)
  model.eval()
  model = model.to(device)

  img_array = np.array(image)
  if model_kind=='real_sr':
    border = 0
    window_size = 8
    img_gt = None
    #st.write(img_array)
    img_lq = cv2.cvtColor(img_array, cv2.IMREAD_COLOR).astype(np.float32) / 255.
  elif model_kind=='lightweight_sr':
    border = 4
    window_size = 8
    img_gt = cv2.cvtColor(img_array, cv2.IMREAD_COLOR).astype(np.float32) / 255.
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
      print(output.shape)
  # save image
  output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
  if output.ndim == 3:
      output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
  output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
  #cv2.imwrite(f'{save_dir}/{imgname}_SwinIR.png', output)

  # evaluate psnr/ssim/psnr_b
  psnr, ssim, psnr_y, ssim_y = 0, 0, 0, 0
  # if img_gt is not None:
  #     img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
  #     img_gt = img_gt[:h_old * 4, :w_old * 4, ...]  # crop gt
  #     img_gt = np.squeeze(img_gt)
  #     img_gt = img_gt.reshape(output.shape)
  #     psnr = util.calculate_psnr(output, img_gt, crop_border=border)
  #     ssim = util.calculate_ssim(output, img_gt, crop_border=border)

  #     if img_gt.ndim == 3:  # RGB image
  #         psnr_y = util.calculate_psnr(output, img_gt, crop_border=border, test_y_channel=True)
  #         ssim_y = util.calculate_ssim(output, img_gt, crop_border=border, test_y_channel=True)

  return output, (psnr, ssim, psnr_y, ssim_y)

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
    
@st.cache
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

  uploaded_file1 = st.file_uploader("Choose an image...", type="jpg",key="SR")
  if uploaded_file1 is not None:
      image = Image.open(uploaded_file1)
      st.image(image, caption='Original Image', use_column_width=False)
      st.write("")
      new_image,score = predict_SR(image,model_kind='lightweight_sr')
      st.image(new_image, caption='Processed Image', use_column_width=True)
      st.write("(psnr, ssim, psnr_y, ssim_y) = ",score)
  ### Deblur ###
  st.title("Deblur")
  st.write("Blurry 이미지를 보정해줍니다.")

  uploaded_file2 = st.file_uploader("Choose an image...", type="jpg",key="DB")
  if uploaded_file2 is not None:
      image = Image.open(uploaded_file2)
      st.image(image, caption='Original Image', use_column_width=False)
      st.write("")
      new_image = predict_DB(image)
      st.image(new_image, caption='Processed Image', use_column_width=True)

if __name__ == '__main__':
	main()