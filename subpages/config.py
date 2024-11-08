import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import cv2
from PIL import Image, ImageOps
import numpy as np

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

nrow = st.number_input("Number of rows", min_value=1, max_value=5, value=5)
st.session_state.nrow = nrow

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

        st.session_state.camera_orientation = "Landscape" if st.session_state.org_picture.shape[1] > st.session_state.org_picture.shape[0] else "Portrait"
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
        
        if st.session_state.camera_orientation == "Portrait":
            w = 400
            h = 640
        else:
            w = 640
            h = 400

        srcs = np.array([[int(x*st.session_state.org_picture.shape[1]), int(y*st.session_state.org_picture.shape[0])] for x, y in st.session_state.points]).astype(np.int32)
        srcs = srcs.reshape((-1, 1, 2))
        dsts = np.array([[0, 0], [w, 0], [w, h], [0, h]]).astype(np.int32)

        if len(srcs) == 4:
            M = cv2.getPerspectiveTransform(srcs.astype(np.float32), dsts.astype(np.float32))
            res_img = cv2.warpPerspective(st.session_state.org_picture, M, (w, h))
            st.session_state.configured = True
            st.session_state.template = res_img
            st.rerun()

if st.session_state.configured and not st.session_state.is_apply:
    st.image(st.session_state.picture, use_column_width=True)

if apply_btn and st.session_state.configured:
    st.session_state.is_apply = True
    st.rerun()

if st.session_state.is_apply and st.session_state.template is not None:
    tmp = st.session_state.template.copy()
    if st.session_state.camera_orientation == "Portrait":
        tmp = cv2.rotate(st.session_state.template, cv2.ROTATE_90_CLOCKWISE)

    h, w = tmp.shape[:2]
    row_images = []
    interval = h // nrow
    for i in range(nrow):
        img = tmp[i*interval:(i+1)*interval, :]
        row_images.append(img)
        st.image(img, use_column_width=True)
    st.session_state.row_images = row_images