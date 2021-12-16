from fastapi import FastAPI, Response, UploadFile, File
from starlette.responses import StreamingResponse
import uvicorn
from PIL import Image
import io
import sys
import os
sys.path.append(os.path.join(os.getcwd(), '../Utils'))
ImageEncoder = __import__("ImageEncoder")
from Wrapper import Deblur

app = FastAPI()
db_predictor = Deblur()

@app.post('/deblur')
async def predict_deblur(image:UploadFile=File(...)):
    image_bytes = await image.read()
    image = ImageEncoder.Decode(image_bytes, channels=3)
    new_image = db_predictor.predict(image)
    img_byte = ImageEncoder.Encode(new_image)
    return Response(content=img_byte)

if __name__ == "__main__":
    uvicorn.run(app="Server:app",
                host="0.0.0.0",
                port=8000,
                reload=True,
                workers=4)