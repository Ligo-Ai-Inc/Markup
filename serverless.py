import runpod  
import base64
import io
from PIL import Image
import numpy as np
import cv2
from models import Processor

processor = Processor("/workspace/checkpoints/sam2.1_hiera_large.pt", \
                        "configs/sam2.1/sam2.1_hiera_l.yaml")

# Main entry point for the serverless function
def main_handler(event):
    """
    Entry point triggered by RunPod.
    """
    input = event['input']
    image_data = input['image_data']
    nrow = input['nrow']

    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    numpy_image = np.array(image)

    output_data, row_rects, row_polygons = processor.process(numpy_image, nrow)
    
    for row, v in row_polygons.items():
        for i, polygon in enumerate(v):
            row_polygons[row][i] = polygon.tolist()
    
    return {
        "output_data": output_data,
        "row_rects": row_rects,
        "row_polygons": row_polygons
    }

if __name__ == '__main__':
    runpod.serverless.start({'handler': main_handler})