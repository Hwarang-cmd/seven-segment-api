# main.py
from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import tempfile
from ssocr import load_image, preprocess, find_digits_positions, recognize_digits_line_method

app = FastAPI()

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    contents = await image.read()
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    blurred, gray_img = load_image(tmp_path)
    dst = preprocess(blurred, 35)
    digits_positions = find_digits_positions(dst)
    digits = recognize_digits_line_method(digits_positions, blurred, dst)

    return {"digits": digits}
