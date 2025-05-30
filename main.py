# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import tempfile
from ssocr import preprocess, find_digits_positions, recognize_digits_line_method, THRESHOLD

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        blurred = cv2.GaussianBlur(img, (7, 7), 0)
        dst = preprocess(blurred, THRESHOLD)
        positions = find_digits_positions(dst)
        digits = recognize_digits_line_method(positions, blurred.copy(), dst)

        return {"digits": digits}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
