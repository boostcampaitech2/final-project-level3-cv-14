#!/bin/sh

python download-weights.py
cd ..

mv Deblur/model_deblurring.pth Deblur/MPRNet/Deblurring/pretrained_models