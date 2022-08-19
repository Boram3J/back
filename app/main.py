import io

import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from starlette.responses import StreamingResponse

from app.ocr_port import run_ocr

app = FastAPI()


@app.get("/")
def root():
    print("ajsdhldhad")
    return {"message": "Hello World"}


@app.post("/api/image")
def get_image(file: UploadFile = File(...)):
    image = np.array(Image.open(file.file))
    # process image
    # convert to png bytes
    return StreamingResponse(io.BytesIO(image.tobytes()), media_type="image/png")


@app.post("/api/translate")
def translate(file: UploadFile = File(...)):
    image = np.array(Image.open(file.file))
    # process image

    # detection
    # recognition
    run_ocr(img=image)
    # papago
    papago()
    # reult
    # convert to png bytes
    return StreamingResponse(io.BytesIO(result.tobytes()), media_type="image/png")
