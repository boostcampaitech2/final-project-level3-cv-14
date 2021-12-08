import os
import argparse
import tensorflow as tf
import torch
import numpy as np
from .model import DEBLUR

class Predictor():
    def __init__(self):
        parser = argparse.ArgumentParser(description='deblur arguments')
        parser.add_argument('--phase', type=str, default='test', help='determine whether train or test')
        parser.add_argument('--model', type=str, default='color', help='model type: [lstm | gray | color]')
        parser.add_argument('--batch_size', help='training batch size', type=int, default=16)
        parser.add_argument('--epoch', help='training epoch number', type=int, default=4000)
        parser.add_argument('--lr', type=float, default=1e-4, dest='learning_rate', help='initial learning rate')
        parser.add_argument('--gpu', dest='gpu_id', type=str, default='0', help='use gpu or cpu')
        parser.add_argument('--height', type=int, default=720,
                            help='height for the tensorflow placeholder, should be multiples of 16')
        parser.add_argument('--width', type=int, default=1280,
                            help='width for the tensorflow placeholder, should be multiple of 16 for 3 scales')
        self.args = parser.parse_args()


    def predict(self,image):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        deblur = DEBLUR(self.args)
        output = deblur.test(self.args.height, self.args.width, np.array(image))
        return output