import os
#import argparse
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from skimage import img_as_ubyte
from collections import OrderedDict
import cv2
import numpy as np
import streamlit as st
import sys
from runpy import run_path
from PIL import Image
sys.path.append(os.path.join(os.getcwd(),'Deblur/MPRNet'))

class Deblur():
    def __init__(self):
        
        task = "Deblurring"
        model_path = "/opt/ml/final-project-level3-cv-14/Deblur/MPRNet/Deblurring"
        
        load_file = run_path(os.path.join(model_path, "MPRNet.py"))
        self.model = load_file['MPRNet']()
        self.model.cuda()

        weights = os.path.join(model_path, "pretrained_models", "model_"+task.lower()+".pth")
        checkpoint = torch.load(weights)
        try:
            self.model.load_state_dict(checkpoint["state_dict"])
        except:
            state_dict = checkpoint["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)
        self.model.eval()
    
    def predict(self, image):

        img_multiple_of = 8

        #img = Image.open(image).convert('RGB')
        #img = Image.open(image)
        img = image
        input_ = TF.to_tensor(img).unsqueeze(0).cuda()

        # Pad the input if not_multiple_of 8
        h,w = input_.shape[2], input_.shape[3]
        H,W = ((h+img_multiple_of)//img_multiple_of)*img_multiple_of, ((w+img_multiple_of)//img_multiple_of)*img_multiple_of
        padh = H-h if h%img_multiple_of!=0 else 0
        padw = W-w if w%img_multiple_of!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

        with torch.no_grad():
            restored = self.model(input_)
        restored = restored[0]
        restored = torch.clamp(restored, 0, 1)

        # Unpad the output
        restored = restored[:,:,:h,:w]

        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])
            
        #cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return restored
            
