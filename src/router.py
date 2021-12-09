from fastapi import FastAPI, APIRouter, UploadFile, File
from starlette.responses import StreamingResponse
from fastapi.responses import FileResponse
import uvicorn
from PIL import Image
import io
import base64
from SuperRes.predict import Predictor as SRPredictor
from Deblur.predict import Predictor as DBPredictor

app = FastAPI()
sr_predictor = SRPredictor()
db_predictor = DBPredictor()

@app.get('/hi')
def hello():
    return {"hello":"world"}

@app.post('/super')
async def predict_super(files:UploadFile=File(...)):
    image_bytes = await files.read()
    image = Image.open(io.BytesIO(image_bytes))
    new_image = sr_predictor.predict(image)
    new_image = Image.fromarray(new_image.astype('uint8'), 'RGB')

    img_byte = io.BytesIO()
    new_image.save(img_byte,"JPEG")
    img_byte.seek(0)

    return StreamingResponse(img_byte,media_type="image/jpeg")

@app.post('/deblur')
async def predict_deblur(files:UploadFile=File(...)):
    image_bytes = await files.read()
    image = Image.open(io.BytesIO(image_bytes))
    new_image = db_predictor.predict(image)
    new_image = Image.fromarray(new_image.astype('uint8'), 'RGB')

    img_byte = io.BytesIO()
    new_image.save(img_byte,"JPEG")
    img_byte.seek(0)

    return StreamingResponse(img_byte,media_type="image/jpeg")

if __name__ == "__main__":
    uvicorn.run(app="router:app",
                host="0.0.0.0",
                port=8000, #TODO: change into remote server port
                reload=True,
                workers=4)