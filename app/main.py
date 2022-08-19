import io

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from starlette.responses import StreamingResponse

from app.ocr_port import run_ocr_and_translate

app = FastAPI()


@app.get("/")
def root():
    print("ajsdhldhad")
    return {"message": "Hello World"}


@app.post("/api/ocr")
def get_image(file: UploadFile = File(...)):
    image = np.array(Image.open(file.file))
    image = run_ocr_and_translate(image)
    _, image = cv2.imencode(".png", image)  # pylint: disable=no-member
    return StreamingResponse(io.BytesIO(image.tobytes()), media_type="image/png")
