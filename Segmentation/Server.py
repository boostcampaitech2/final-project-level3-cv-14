'''
작성자 홍지연, 김지성
최종 수정일 2021-12-09
'''

import sys, os
from fastapi import FastAPI, File, Response
import uvicorn
from Wrapper import PeopleSegmentation
sys.path.append(os.path.join(os.getcwd(), '../Utils'))
ImageEncoder = __import__("ImageEncoder")

peopleSeg = PeopleSegmentation('cuda')
app = FastAPI()
@app.post("/inference/")
async def inference(image: bytes = File(...), mask: bytes = File(...)):
    image = ImageEncoder.Decode(image, channels=1)
    masks = peopleSeg.inference(image)
    masks_bytes = ImageEncoder.Encode(masks, ext='jpg', quality=90)
    return Response(content=masks_bytes)

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8786)