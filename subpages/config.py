import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import cv2
from PIL import Image, ImageOps
import numpy as np
# from models import Processor
import yaml
import os
import requests

if "points" not in st.session_state:
    st.session_state.points = []
if "prev_point" not in st.session_state:
    st.session_state.prev_point = None
if "configured" not in st.session_state:
    st.session_state.configured = False
if "template" not in st.session_state:
    st.session_state.template = None
if "picture" not in st.session_state:
    st.session_state.picture = None
if "org_picture" not in st.session_state:
    st.session_state.org_picture = None
if "is_apply" not in st.session_state:
    st.session_state.is_apply = False
if "row_segments" not in st.session_state:
    st.session_state.row_segments = None
if "config" not in st.session_state:
    st.session_state.config = None

def load_config():
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r") as f:
            st.session_state.config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        st.session_state.config = {}

def save_config():
    st.session_state.config['nrow'] = st.session_state.nrow
    with open("config.yaml", "w") as f:
        yaml.dump(st.session_state.config, f)

load_config()
st.number_input("Number of rows", min_value=1, max_value=10, value=st.session_state.config.get("nrow", 5), key="nrow", on_change=save_config)

url = "http://localhost:8000/sam"
if os.path.exists("tmp.txt"):
    with open("tmp.txt", "r") as f:
        url = f.read()
# if "processor" not in st.session_state:
#     st.session_state.processor = Processor("../sam2/checkpoints/sam2.1_hiera_large.pt", \
#                                            "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml")

configure = st.button("Reset")
apply_btn = st.button("Apply")

if configure:
    st.session_state.configured = False
    st.session_state.template = None
    st.session_state.points = []
    st.session_state.picture = None
    st.session_state.org_picture = None
    st.session_state.camera_orientation = None
    st.session_state.is_apply = False
    st.rerun()

if not st.session_state.configured:

    img_file_buffer = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if img_file_buffer is not None:
        st.write("Click the four corners of the box in the image.")
        if st.session_state.picture is None:
            org_picture = Image.open(img_file_buffer)
            org_picture = ImageOps.exif_transpose(org_picture)
            data = list(org_picture.getdata())
            org_picture_without_exif = Image.new(org_picture.mode, org_picture.size)
            org_picture_without_exif.putdata(data)
            org_picture = org_picture_without_exif
            org_picture = np.array(org_picture)
            st.session_state.org_picture = org_picture

        picture = st.session_state.org_picture.copy()
        scale_factor = 1080 / st.session_state.org_picture.shape[1]
        picture = cv2.resize(picture, (1080, int(picture.shape[0] * scale_factor)))
        st.session_state.picture = picture

        # st.session_state.camera_orientation = "Landscape" if st.session_state.org_picture.shape[1] > st.session_state.org_picture.shape[0] else "Portrait"
        vis_points = np.array([[int(x*st.session_state.picture.shape[1]), int(y*st.session_state.picture.shape[0])] for x, y in st.session_state.points]).astype(np.int32)
        vis_points = vis_points.reshape((-1, 1, 2))
        if len(vis_points) > 1:
            st.session_state.picture = cv2.polylines(st.session_state.picture, [vis_points], isClosed=True, color=(0, 255, 0), thickness=10)

        value = streamlit_image_coordinates(
            st.session_state.picture,
            use_column_width=True,
        )

        if value is not None:
            x = value['x'] / value['width']
            y = value['y'] / value['height']
            if [x, y] != st.session_state.prev_point:
                st.session_state.prev_point = [x, y]
                st.session_state.points.append([x, y])
                st.rerun()

        srcs = np.array([[int(x*st.session_state.org_picture.shape[1]), int(y*st.session_state.org_picture.shape[0])] for x, y in st.session_state.points]).astype(np.int32)
        srcs = srcs.reshape((-1, 1, 2))

        if len(srcs) == 4:
            x, y, w, h = cv2.boundingRect(srcs)
            st.session_state.camera_orientation = "Landscape" if w > h else "Portrait"
            if st.session_state.camera_orientation == "Portrait":
                w = int(400 / 5 * nrow)
                h = 640
            else:
                w = 640
                h = int(400 / 5 * nrow)
            dsts = np.array([[0, 0], [w, 0], [w, h], [0, h]]).astype(np.int32)
            M = cv2.getPerspectiveTransform(srcs.astype(np.float32), dsts.astype(np.float32))
            res_img = cv2.warpPerspective(st.session_state.org_picture, M, (w, h))
            st.session_state.configured = True
            st.session_state.template = res_img
            st.rerun()

if st.session_state.configured and not st.session_state.is_apply:
    st.image(st.session_state.picture, use_container_width=True)

if apply_btn and st.session_state.configured:
    st.session_state.is_apply = True
    st.rerun()

if st.session_state.is_apply and st.session_state.template is not None:
    template = st.session_state.template.copy()
    if st.session_state.camera_orientation == "Portrait":
        template = cv2.rotate(st.session_state.template, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite("template.png", cv2.cvtColor(template, cv2.COLOR_RGB2BGR))

    # output_data, row_rects, row_polygons = st.session_state.processor.process(template, st.session_state.nrow)
    payload = {'nrow': st.session_state.nrow}
    files=[
    ('file',('template.png',open('template.png','rb'),'image/jpeg'))
    ]
    headers = {}

    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    response = response.json()
    output_data = response['output_data']
    row_rects = response['row_rects']
    row_polygons = response['row_polygons']

    full_mask = np.zeros_like(template)
    for i, row in enumerate(row_polygons):
        for poly in row_polygons[row]:
            random_color = np.random.randint(0, 256, size=3, dtype=np.uint8)
            random_color = [int(x) for x in random_color]
            cv2.fillPoly(full_mask, [np.array(poly).reshape((-1, 1, 2))], random_color)
    template = cv2.addWeighted(template, 0.7, full_mask, 0.4, 0)

    row_images = []
    for i, row in enumerate(row_rects):
        x, y, w, h = row
        x = max(0, x)
        y = max(0, y)
        row_images.append(template[y:y+h, x:x+w])

    st.session_state.row_images = row_images
    st.session_state.row_segments = output_data
    for i, row in enumerate(row_images):
        st.image(row, use_container_width=True)
    st.session_state.configured = False
    st.session_state.is_apply = False