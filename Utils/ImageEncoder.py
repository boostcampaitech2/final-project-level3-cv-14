'''
이 모듈은 다음과 같은 기능을 제공합니다.
 - Encode : numpy type 이미지를 bytes type으로 serialize합니다.
 - Decode : bytes typq으로 serialize된 이미지를 numpy type으로 복원합니다.

작성자 김지성
최종 수정일 2021-12-09
'''

import cv2
import numpy as np

def Encode(image, ext='jpg', quality=90):
    if ext == 'jpg':
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, image_serial = cv2.imencode('.jpg', image, encode_param)
    elif ext == 'png':
        _, image_serial = cv2.imencode('.png', image)
    image_bytes = image_serial.tobytes()
    return image_bytes

def Decode(image_bytes, channels=3):
    if channels==3:
        color = cv2.IMREAD_COLOR
    elif channels==1:
        color = cv2.IMREAD_GRAYSCALE

    image_serial = np.fromstring(image_bytes, np.uint8)
    image = cv2.imdecode(image_serial, color)
    return image