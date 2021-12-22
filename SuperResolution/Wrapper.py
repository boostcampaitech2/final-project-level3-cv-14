import os
import torch
import numpy as np
import streamlit as st
import cv2
import sys
sys.path.append(os.path.join(os.getcwd(),'SwinIR'))
from SwinIR.models.network_swinir import SwinIR as net


class SuperResolution():
    def __init__(self):
        model_path = os.path.join(os.getcwd(),'SwinIR/experiments/pretrained_models/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth')
 
        self.model = net(upscale=4, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        pretrained_model = torch.load(model_path)
        self.model.load_state_dict(pretrained_model['params_ema'] if 'params_ema' in pretrained_model.keys() else pretrained_model, strict=True)
        self.model.to(self.device).eval()


    @st.cache
    def predict(self, image):
        # setup window_size
        window_size = 8
        
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
