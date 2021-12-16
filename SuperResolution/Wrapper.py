import os
import argparse
import tensorflow as tf
import torch
import numpy as np
import streamlit as st
import cv2
import sys
sys.path.append(os.path.join(os.getcwd(),'SwinIR'))
from SwinIR.main_test_swinir import define_model, setup, get_image_pair

class SuperResolution():
    def __init__(self):
        model_dir = os.path.join(os.getcwd(),'SwinIR/experiments/pretrained_models')

        self.model_zoo = {
            'real_sr': {
                4: os.path.join(model_dir, '003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth')
            },
            'classical_sr':{
                2: os.path.join(model_dir, '001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth'),
                3: os.path.join(model_dir, '001_classicalSR_DF2K_s64w8_SwinIR-M_x3.pth'),
                4: os.path.join(model_dir, '001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth'),
                8: os.path.join(model_dir, '001_classicalSR_DF2K_s64w8_SwinIR-M_x8.pth'),
            }
        }
 
        args = argparse.Namespace()
        args.folder_gt = None
        args.folder_lq = None
        args.large_model = None
        
        args.task = 'classical_sr'
        args.noise = 15
        args.jpeg = 40
        args.training_patch_size = 64
        args.scale = 2
        args.model_path = self.model_zoo[args.task][args.scale]

        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tasks = {
            'Real-World Image Super-Resolution': 'real_sr',
            'Grayscale Image Denoising': 'gray_dn',
            'Color Image Denoising': 'color_dn',
            'JPEG Compression Artifact Reduction': 'jpeg_car'
        }

        self.model = define_model(self.args).to(self.device).eval()

    def change_scale(self, scale):
        if self.args.scale != scale:
            self.args.scale = scale
            if scale == 4:
                self.args.task = 'real_sr'
                self.args.training_patch_size = 128
                self.args.model_path = self.model_zoo[self.args.task][self.args.scale]
            else:
                self.args.task = 'classical_sr'
                self.args.training_patch_size = 64
                self.args.model_path = self.model_zoo[self.args.task][self.args.scale]
            self.model = define_model(self.args).to(self.device).eval()
        
        
    @st.cache
    def predict(self, image, task_type='Real-World Image Super-Resolution', jpeg=40, noise=15,scale=4):
        if self.args.scale != scale:
            self.change_scale(scale)

        # setup folder and path
        _, _, border, window_size = setup(self.args)
        
        # read image
        img_gt = cv2.cvtColor(image, cv2.IMREAD_COLOR).astype(np.float32) / 255. # image to HWC-BGR, float32
        img_gt = np.transpose(img_gt if img_gt.shape[2] == 1 else img_gt[:, :, [2, 1, 0]],
                                (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_gt = torch.from_numpy(img_gt).float().unsqueeze(0).to(self.device)  # CHW-RGB to NCHW-RGB

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_gt.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_gt = torch.cat([img_gt, torch.flip(img_gt, [2])], 2)[:, :, :h_old + h_pad, :]
            img_gt = torch.cat([img_gt, torch.flip(img_gt, [3])], 3)[:, :, :, :w_old + w_pad]
            output = self.model(img_gt)
            output = output[..., :h_old * 4, :w_old * 4]
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)
        return output
