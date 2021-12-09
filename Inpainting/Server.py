'''
작성자 김지성
최종 수정일 2021-12-09
'''

import sys, os
from fastapi import FastAPI, File, Response
import uvicorn
from Wrapper import LaMa
sys.path.append(os.path.join(os.getcwd(), '../Utils'))
ImageEncoder = __import__("ImageEncoder")

lama = LaMa('cuda')
app = FastAPI()
@app.post("/inference/")
async def inference(image: bytes = File(...), mask: bytes = File(...)):
    image = ImageEncoder.Decode(image, channels=3)
    mask = ImageEncoder.Decode(mask, channels=1)
    result = lama.inference(image, mask)
    image_bytes = ImageEncoder.Encode(result, ext='jpg', quality=90)
    return Response(content=image_bytes)

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8786)