from fastapi import FastAPI
from fastapi import File, UploadFile, Form
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import cv2

from models import Processor

processor = Processor("/workspace/checkpoints/sam2.1_hiera_large.pt", \
                        "configs/sam2.1/sam2.1_hiera_l.yaml")

app = FastAPI()

@app.post("/sam")
async def sam(file: UploadFile = File(...), nrow: int = Form(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    numpy_image = np.array(image)
    numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    output_data, row_rects, row_polygons = processor.process(numpy_image, nrow)

    for row, v in row_polygons.items():
        for i, polygon in enumerate(v):
            row_polygons[row][i] = polygon.tolist()
    
    return JSONResponse(content={
        "output_data": output_data,
        "row_rects": row_rects,
        "row_polygons": row_polygons
    })