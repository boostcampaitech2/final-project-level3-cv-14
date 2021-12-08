import os
import streamlit as st
import PIL.Image as Image

def button(output_img, save_dir):
        """ 아웃풋 이미지를 다운로드받는 버튼 생성
        
        Keyword arguments:
        output_img : PIL image
        save_dir : 이미지를 저장할 디렉토리

        """
        IMAGE_SAVE_PATH = os.path.join(save_dir, 'result.jpg')
        output_img.save(IMAGE_SAVE_PATH, format="JPEG")
        with open(IMAGE_SAVE_PATH, "rb") as fp:
            btn = st.download_button(
                label="Download Your Image",
                data=fp,
                file_name='result.jpg',
                mime="image/jpeg",
            )
            