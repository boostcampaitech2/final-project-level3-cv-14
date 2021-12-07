import albumentations as albu
import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from iglovikov_helper_functions.utils.image_utils import pad, unpad
from people_segmentation.pre_trained_models import create_model

MAX_SIZE = 512 # 이외 사이즈 실험

@st.cache(allow_output_mutation=True)
def cached_model():
    model = create_model("Unet_2020-07-20")
    model.eval()
    return model

model = cached_model()
transform = albu.Compose(
    [albu.LongestMaxSize(max_size=MAX_SIZE), albu.Normalize(p=1)], p=1
)

st.title("Segment people")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    original_image = np.array(Image.open(uploaded_file)) #(h, w, c)
    st.image(original_image, caption="Before", use_column_width=True)
    st.write("")
    st.write("Detecting people...")

    original_height, original_width = original_image.shape[:2]
    image = transform(image=original_image)["image"]
    padded_image, pads = pad(image, factor=MAX_SIZE, border=cv2.BORDER_CONSTANT) # image size가 factor로 divisible하도록, pads =(x_min_pad, y_min_pad, x_max_pad, y_max_pad)

    x = torch.unsqueeze(tensor_from_rgb_image(padded_image), 0) # (1, c, h, w)

    with torch.no_grad():
        prediction = model(x)[0][0]

    mask = (prediction > 0).cpu().numpy().astype(np.uint8)
    mask = unpad(mask, pads) #Crops patch from the center so that sides are equal to pads.
    mask = cv2.resize(
        mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST
    ) 
    mask_3_channels = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    dst = cv2.addWeighted(
        original_image, 1, (mask_3_channels * (0, 255, 0)).astype(np.uint8), 0.5, 0
    ) #imgA * a + imgB * b + c , a+b=1이면 알파블렌딩
    
    # split result
    # mask = np.uint8((mask > 127) * 255)
    cnt, labels = cv2.connectedComponents(mask)
    h,w = mask.shape
    result = np.zeros((h,w,3), dtype=np.uint8)
    for i in range(1, cnt):
        result[labels==i] = [int(j) for j in np.random.randint(0,255, 3)]
        
    st.image(mask * 255, caption="Mask", use_column_width=True)
    st.image(dst, caption="Image + mask", use_column_width=True)
    st.image(result, caption="Split Mask", use_column_width=True)
    
    # select region
    selected_region = st.radio("Select Num", range(1, cnt+1))
    selected_result = np.zeros((h,w,3), dtype=np.uint8)
    selected_result[labels==selected_region] =  [int(j) for j in np.random.randint(0,255, 3)]
    st.image(selected_result, caption="Split Mask", use_column_width=True)
