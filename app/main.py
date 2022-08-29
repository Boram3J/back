import io
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from starlette.responses import StreamingResponse

from app.ocr_port import run_ocr_and_translate
from app.text import inpaint_text

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Hello World"}


@app.post("/api/text")
def text(file: UploadFile = File(...)):
    image = np.array(Image.open(file.file))
    image = inpaint_text(image)
    _, image = cv2.imencode(".png", image)  # pylint: disable=no-member
    return StreamingResponse(io.BytesIO(image.tobytes()), media_type="image/png")


@app.post("/api/translate")
def translate(file: UploadFile = File(...)):
    image = np.array(Image.open(file.file))
    image = run_ocr_and_translate(image)
    _, image = cv2.imencode(".png", image)  # pylint: disable=no-member
    return StreamingResponse(io.BytesIO(image.tobytes()), media_type="image/png")
