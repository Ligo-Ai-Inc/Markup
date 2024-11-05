import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import cv2
from PIL import Image
import numpy as np
from camera_input_live import camera_input_live

if "points" not in st.session_state:
    st.session_state.points = []
if "prev_point" not in st.session_state:
    st.session_state.prev_point = None
if "configured" not in st.session_state:
    st.session_state.configured = False
if "template" not in st.session_state:
    st.session_state.template = None

nrow = st.number_input("Number of rows", min_value=1, max_value=5, value=5)
st.session_state.nrow = nrow

configure = st.button("Configure")

camera_orientation = st.radio("Camera Orientation", ["Landscape", "Portrait"])

if configure:
    st.session_state.configured = False
    st.session_state.template = None
    st.session_state.points = []

if not st.session_state.configured:

    img_file_buffer = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if img_file_buffer is not None:
        org_picture = Image.open(img_file_buffer)
        org_picture = np.array(org_picture)
        picture = org_picture.copy()

        srcs = np.array([[int(x*picture.shape[1]), int(y*picture.shape[0])] for x, y in st.session_state.points]).astype(np.int32)
        srcs = srcs.reshape((-1, 1, 2))
        if len(srcs) > 1:
            picture = cv2.polylines(picture, [srcs], isClosed=True, color=(0, 255, 0), thickness=10)

        value = streamlit_image_coordinates(
            picture,
            use_column_width=True,
        )

        if value is not None:
            x = value['x'] / value['width']
            y = value['y'] / value['height']
            if [x, y] != st.session_state.prev_point:
                st.session_state.prev_point = [x, y]
                st.session_state.points.append([x, y])
                st.rerun()
        
        if camera_orientation == "Portrait":
            w = 400
            h = 640
        else:
            w = 640
            h = 400
        dsts = np.array([[0, 0], [w, 0], [w, h], [0, h]]).astype(np.int32)

        if len(srcs) == 4:
            M = cv2.getPerspectiveTransform(srcs.astype(np.float32), dsts.astype(np.float32))
            res_img = cv2.warpPerspective(org_picture, M, (w, h))
            st.session_state.configured = True
            st.session_state.template = res_img
            st.rerun()

if st.session_state.template is not None:
    # st.image(st.session_state.template, use_column_width=True)
    h, w = st.session_state.template.shape[:2]
    row_images = []
    interval = h // nrow
    for i in range(nrow):
        img = st.session_state.template[i*interval:(i+1)*interval, :]
        row_images.append(img)
        st.image(img, use_column_width=True)
    st.session_state.row_images = row_images