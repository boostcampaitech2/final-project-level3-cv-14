from fastapi import FastAPI, UploadFile, File, Response
import uvicorn
from PIL import Image
import io
import sys
sys.path.append('/opt/ml/final-project/')
from Utils import ImageEncoder
from Wrapper import SuperResolution

app = FastAPI()
sr_predictor = SuperResolution()

@app.post('/super')
async def predict_super(files:UploadFile=File(...)):
    image_bytes = await files.read()
    image = ImageEncoder.Decode(image_bytes, channels=3)
    new_image = sr_predictor.predict(image)
    img_byte = ImageEncoder.Encode(new_image)
    return Response(content=img_byte)

if __name__ == "__main__":
    uvicorn.run(app="Server:app",
                host="0.0.0.0",
                port=8000, #TODO: change into remote server port
                reload=True,
                workers=4)