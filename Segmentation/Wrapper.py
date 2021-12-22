'''
작성자 홍지연, 김지성
최종 수정일 2021-12-09
'''

from people_segmentation.pre_trained_models import create_model
import torch
import numpy as np
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from iglovikov_helper_functions.utils.image_utils import pad, unpad
import albumentations as alb
import cv2

class PeopleSegmentation:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = create_model("Unet_2020-07-20")
        self.model.to(device)
        self.model.eval()



    def inference(self, image):
        h,w = image.shape[:2]
        image = alb.LongestMaxSize(max_size=1024, p=1)(image=image)["image"]
        image = alb.Normalize(p=1)(image=image)["image"]
        image, pads = pad(image, factor=1024, border=cv2.BORDER_CONSTANT)
        image = torch.unsqueeze(tensor_from_rgb_image(image), 0)
        image = image.to(self.device)

        with torch.no_grad():
            masks = self.model(image)[0][0]

        masks = (masks > 0).cpu().numpy().astype(np.uint8)
        masks = unpad(masks, pads)  # Crops patch from the center so that sides are equal to pads.
        masks = cv2.resize(masks, (w, h), interpolation=cv2.INTER_NEAREST)
        return masks*255