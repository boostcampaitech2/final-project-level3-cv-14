from fastapi import FastAPI, UploadFile, File, Response
import uvicorn
import sys
import os
sys.path.append(os.path.join(os.getcwd(), '../Utils'))
ImageEncoder = __import__("ImageEncoder")
from Wrapper import SuperResolution


app = FastAPI()
sr_predictor = SuperResolution()


@app.post('/super')
async def predict_super(image:UploadFile=File(...)):
    image_bytes = await image.read()
    image = ImageEncoder.Decode(image_bytes, channels=3)
    new_image = sr_predictor.predict(image)
    img_byte = ImageEncoder.Encode(new_image)
    return Response(content=img_byte)


if __name__ == "__main__":
    uvicorn.run(app="Server:app",
                host="0.0.0.0",
                port=6006,
                reload=True,
                workers=4)
