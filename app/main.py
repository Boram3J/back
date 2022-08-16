import io

import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from starlette.responses import StreamingResponse

app = FastAPI()


@app.post("/image")
def get_image(file: UploadFile = File(...)):
    image = np.array(Image.open(file.file))
    # process image
    # convert to png bytes
    return StreamingResponse(io.BytesIO(image.tobytes()), media_type="image/png")
