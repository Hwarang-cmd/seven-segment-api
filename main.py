from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import tempfile
import ssocr

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # เรียกใช้ ssocr
        blurred, gray_img = ssocr.load_image(tmp_path, show=False)
        dst = ssocr.preprocess(blurred, ssocr.THRESHOLD, show=False)
        digits_positions = ssocr.find_digits_positions(dst)
        digits = ssocr.recognize_digits_line_method(digits_positions, blurred, dst)

        return JSONResponse(content={"digits": digits})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
