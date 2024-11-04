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

# configure = st.button("Configure")
# show_btn = st.button("Show")

# if show_btn:
#     st.session_state.is_show = not st.session_state.is_show

# if st.session_state.is_show:
#     if st.session_state.template is not None:
#         img_file_buffer = st.camera_input("Take a picture")

# if configure:
#     st.session_state.configured = False

if not st.session_state.configured:
    image = camera_input_live()
    img_file_buffer = st.empty()
    while True:
        img_file_buffer.image(image, use_column_width=True)
        if st.button("Capture"):
            break

    # img_file_buffer = st.camera_input("Take a picture")
    # img_file_buffer = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # if img_file_buffer is not None:
    #     org_picture = Image.open(img_file_buffer)
    #     org_picture = np.array(org_picture)
    #     picture = org_picture.copy()

    #     srcs = np.array([[int(x*picture.shape[1]), int(y*picture.shape[0])] for x, y in st.session_state.points]).astype(np.int32)
    #     srcs = srcs.reshape((-1, 1, 2))
    #     if len(srcs) > 1:
    #         picture = cv2.polylines(picture, [srcs], isClosed=True, color=(0, 255, 0), thickness=10)

    #     value = streamlit_image_coordinates(
    #         picture,
    #         use_column_width=True,
    #     )

    #     if value is not None:
    #         x = value['x'] / value['width']
    #         y = value['y'] / value['height']
    #         if [x, y] != st.session_state.prev_point:
    #             st.session_state.prev_point = [x, y]
    #             if len(st.session_state.points) == 4:
    #                 st.session_state.points = []
    #             st.session_state.points.append([x, y])
    #             st.rerun()

    #     dsts = np.array([[0, 0], [400, 0], [400, 640], [0, 640]]).astype(np.int32)

    #     if len(srcs) == 4:
    #         M = cv2.getPerspectiveTransform(srcs.astype(np.float32), dsts.astype(np.float32))
    #         res_img = cv2.warpPerspective(org_picture, M, (400, 640))
    #         st.session_state.configured = True
    #         st.session_state.template = res_img

# if st.session_state.template is not None:
#     st.image(st.session_state.template, use_column_width=True)