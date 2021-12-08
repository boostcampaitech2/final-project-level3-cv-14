import numpy as np
import PIL.Image as Image
import torch
import cv2
import streamlit as st

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../lama'))
from lama.saicinpainting.evaluation.data import pad_img_to_modulo
from lama.saicinpainting.evaluation.utils import move_to_device
from torch.utils.data._utils.collate import default_collate

device = torch.device('cuda')

def inference(img, mask, model, mask_dilation=False,):
    """   
    Keyword arguments:
    img -- PIL image
    mask -- numpy array(unit8). shape : (h, w) 
    model -- inference model
    mask_dilation -- bool, 마스크 영역이 타이트할 때 부풀리는 기능(default:False)
    
    Return :
    output_image -- PIL image after inpainting, (0~255 범위 unit8)

     """
    img = np.array(img.convert('RGB'))
    img = np.transpose(img, (2, 0, 1))
    img = img.astype('float32') / 255
    
    if mask_dilation:
        dilation_kernel = np.ones((3,3), np.uint8) 
        mask = cv2.dilate(mask, dilation_kernel, iterations=15)

    result = dict(image=img, mask=mask[None, ...])
    result['image'] = pad_img_to_modulo(result['image'], 8)
    result['mask'] = pad_img_to_modulo(result['mask'], 8)

    batch = move_to_device(default_collate([result]), device)

    batch['mask'] = (batch['mask'] > 0) * 1 # 0을 넘는 경우는 True이고, True를 1을 곱해 정수로 만들어 0과 1만 가지도록 만듦 (resize 할때 보간법 등이 사용되기 때문에 0~1 사이 값이 생기는 경우가 있어 이런 처리가 필요)
    batch = model(batch)
    cur_res = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()  
    st.image(cur_res, use_column_width='auto', caption="After")

    output_image = Image.fromarray(np.clip(cur_res * 255, 0, 255).astype('uint8')) # 0~1 범위 -> 0~255범위로 조정
    return output_image
