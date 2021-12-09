'''
작성자 홍지연, 김지성
최종 수정일 2021-12-09
'''

from people_segmentation.pre_trained_models import create_model
import torch
import numpy as np

class PeopleSegmentation:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = create_model("Unet_2020-07-20")
        self.model.to(device)
        self.model.eval()


    def inference(self, image):
        with torch.no_grad():
            masks = self.model(image)[0][0]
        masks = (masks > 0).cpu().numpy().astype(np.uint8)
        return masks