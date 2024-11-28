import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from st_clickable_images import clickable_images
import base64
from io import BytesIO
import json
import os
import yaml
import cv2
import numpy as np

measure_btns = []
scan_folder = "scans"
os.makedirs(scan_folder, exist_ok=True)

def load_config():
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r") as f:
            st.session_state.config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        st.session_state.config = {}

load_config()
st.session_state.nrow = st.session_state.config.get("nrow", 5)

if "data" not in st.session_state:
    st.session_state.data = {}
if "pre_loaded" not in st.session_state:
    st.session_state.pre_loaded = False
if "prev_click" not in st.session_state:
    st.session_state.prev_click = None
if "choossen" not in st.session_state:
    st.session_state.choossen = set()
if "row_segments" not in st.session_state:
    st.session_state.row_segments = None
if "prev_row_click" not in st.session_state:
    st.session_state.prev_row_click = [-1] * st.session_state.nrow

list_categories = ["empty", "rock <10cm", "rock >10cm", "crumbly", "block"]
color_map = {
    "empty": "#ff6361",
    "rock <10cm": "#9e61fa",
    "rock >10cm": "#003f5c",
    "crumbly": "#bc5090",
    "block": "#ffa600"
}

def merge_rocks(data):
    for row in data:
        rock_lengths, categories, points_coord = data[row]
        merged_rock_lengths = []
        merged_categories = []
        block_values = []
        merged_points_coord = []
        prev_length = None

        for rock_percentage, category, point in zip(rock_lengths, categories, points_coord):
            rock_length = rock_percentage * st.session_state.box_length
            if rock_length < 1:
                continue

            if category != "empty":
                merged_rock_lengths.append(rock_length)
                merged_categories.append("block")
                block_values.append(float(category))
                merged_points_coord.append(point)
                prev_length = rock_length
                continue

            if len(merged_rock_lengths) == 0 or merged_categories[-1] == "block":
                merged_rock_lengths.append(rock_length)
                merged_points_coord.append(point)
                merged_categories.append("rock <10cm" if rock_length < 10 else "rock >10cm")
                block_values.append(0)
                prev_length = rock_length
                continue

            if rock_length < 10:
                if prev_length < 10:
                    merged_rock_lengths[-1] += rock_length
                    x1, x2 = merged_points_coord[-1]
                    x3, x4 = point
                    minx = min(x1[0], x2[0], x3[0], x4[0])
                    maxx = max(x1[0], x2[0], x3[0], x4[0])
                    y = x1[1]
                    merged_points_coord[-1] = [[minx, y], [maxx, y]]
                else:
                    merged_rock_lengths.append(rock_length)
                    merged_points_coord.append(point)
                    merged_categories.append("rock <10cm")
                    block_values.append(0)
            else:
                merged_rock_lengths.append(rock_length)
                merged_points_coord.append(point)
                merged_categories.append("rock >10cm")
                block_values.append(0)

            prev_length = rock_length

        st.session_state.data[row] = [merged_rock_lengths, merged_categories, merged_points_coord]

def gen_images(rock_lengths, categories, points_coord, row):
    full_width = 1000
    height = 50
    list_images = []

    row_img = st.session_state.row_images[row]
    half_height = row_img.shape[0] // 4
    for j in range(len(points_coord)):
        x1, y1 = points_coord[j][0]
        x2, y2 = points_coord[j][1]
        vx = x2 - x1
        vy = y2 - y1
        perp_vx = -vy
        perp_vy = vx
        random_color = np.random.randint(0, 256, size=3, dtype=np.uint8)
        random_color = [int(x) for x in random_color]

        x1 = np.max([0, x1])

        p1 = (int(x1), int(y1 - half_height * perp_vy))
        p2 = (int(x1), int(y1 + half_height * perp_vy))
        p3 = (int(x2), int(y2 - half_height * perp_vy))
        p4 = (int(x2), int(y2 + half_height * perp_vy))
        # cv2.line(row_img, tuple([int(x1), int(y1)]), tuple([int(x2), int(y2)]), random_color, 2)
        cv2.line(row_img, p1, p2, random_color, 2)
        cv2.line(row_img, p3, p4, random_color, 2)  

    st.session_state.row_images[row] = row_img
        
    for rock_length, category, point in zip(rock_lengths, categories, points_coord):
        rock_percentage = (rock_length / st.session_state.box_length)
        width = int(full_width * rock_percentage)
        image = Image.new("RGB", (width, height), color_map[category])
        draw = ImageDraw.Draw(image)
        font_size = 30
        font = ImageFont.load_default(font_size)
        text = f"{rock_length:.1f}"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = (width - text_width) // 2
        text_y = (height - text_height) // 2
        draw.text((text_x, text_y), text, fill="white", font=font)

        draw.line((0, 0, 0, height), fill="black", width=5)
        draw.line((width-2, 0, width-2, height), fill="black", width=5)

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        html = f"data:image/png;base64,{img_str}"
        list_images.append(html)

    return list_images

@st.dialog("Add new Hole ID")
def add_hole_id():
    id = st.text_input("Enter the hole ID: ")
    submit_btn = st.button("Submit")
    if submit_btn:
        st.session_state.hole_id = id.upper()
        os.makedirs(os.path.join(scan_folder, id.upper()), exist_ok=True)
        st.rerun()

folders = [f for f in os.listdir(scan_folder) if os.path.isdir(os.path.join(scan_folder, f))]
hole_id_list = sorted(folders, key=lambda x: os.path.getctime(os.path.join(scan_folder, x)))
hole_id_idx = len(hole_id_list) - 1
if "hole_id" in st.session_state:
    hole_id_idx = hole_id_list.index(st.session_state.hole_id)
hole_id = st.sidebar.selectbox("Select Hole ID", hole_id_list, index=hole_id_idx)
if hole_id == None:
    st.sidebar.error("No Hole ID found!")
else:
    st.session_state.hole_id = hole_id

st.markdown(f"## Hole ID: {hole_id}")
core_direction = st.radio("Direction", options=["Count Up", "Count Down"], horizontal=True, label_visibility="collapsed")
is_print = st.checkbox("Print barcode", value=False)

st.number_input("Box length", min_value=0, max_value=1000, value=60, key="box_length") # cm
# box_length = st.session_state.box_length
col1, col2 = st.columns(2)
box_from = col1.number_input("Box from", min_value=0.0, max_value=1000.0, value=st.session_state.get("box_from", 0.000))
box_to = col2.number_input("Box to", min_value=0.0, max_value=1000.0, value=st.session_state.get("box_to", 0.000))

# nrow = st.number_input("Number of rows", min_value=1, max_value=5, value=st.session_state.nrow)
# st.session_state.nrow = nrow

add_holeID_btn = st.sidebar.button("Add Hole ID")
sync_btn = st.sidebar.button("Sync")
usb_export = st.sidebar.button("USB Export")
if add_holeID_btn:
    add_hole_id()
st.sidebar.image("assets/labels.png")

if hole_id != None:
    valid = True
    list_scans = sorted(os.listdir(os.path.join(scan_folder, hole_id)), key=lambda x: float(x.split("_")[0]))
    if box_from == box_to:
        valid = False
    else:
        for scan in list_scans:
            item_from = float(scan.split("_")[0])
            item_to = float(scan.split("_")[1])
            if not ((box_from < item_from and box_to <= item_from) or (box_from >= item_to and box_to > item_to)):
                if not (box_from == item_from and box_to == item_to):
                    st.toast("This scan is overlapped with the existing scan!")
                    valid = False
                    break

    if valid:
        if "row_segments" in st.session_state and st.session_state.row_segments is not None:
            os.makedirs(f"{scan_folder}/{hole_id}/{box_from}_{box_to}", exist_ok=True)
            with open(f"{scan_folder}/{hole_id}/{box_from}_{box_to}/save_records.json", "w") as f:
                json.dump(st.session_state.row_segments, f)
                
            for k, v in st.session_state.row_segments.items():
                st.session_state.data[int(k)] = v
            merge_rocks(st.session_state.data)
            st.session_state.row_segments = None
            st.rerun()

        st.session_state.box_from = box_from
        st.session_state.box_to = box_to

        save_btn = st.button("Save")

        list_clicks = []
        for i in range(st.session_state.nrow):
            col1, col2 = st.columns([0.2, 0.8])
            btn = col1.button("Measure row {}".format(i+1))   
            measure_btns.append(btn)

            if i in st.session_state.data:
                rock_lengths, categories, points_coord = st.session_state.data[i]
                list_images = gen_images(rock_lengths, categories, points_coord, i)

                with col2.container(border=False):
                    clicked = clickable_images(
                        list_images,
                        titles=[f"{i}_{j}" for j in range(len(list_images))],
                        div_style={"display": "flex", "justify-content": "space-between", "background-color": "white", "gap": "0px"},
                        img_style={"margin": "0px", "height": "20px"},
                    )
                    list_clicks.append(clicked)

            if "row_images" in st.session_state:
                col2.image(st.session_state.row_images[i], use_container_width=True)

        for row, click in enumerate(list_clicks):
            st.write(f"Row {row+1}: {click}", st.session_state.prev_row_click[row])
            if click != st.session_state.prev_row_click[row]:
                st.session_state.prev_row_click[row] = click
                st.session_state.choossen.add(f"{row}_{click}")

            # if click != -1 and f"{row}_{click}" != st.session_state.prev_click:
                # st.session_state.prev_click = f"{row}_{click}"
                # st.session_state.choossen.add(f"{row}_{click}")

        # def load_measurement_data(measure_data):
        #     lines = measure_data.split("\n")
        #     list_measurement_values = []
        #     for line in lines[:]:
        #         value = float(line.split(":")[-1].strip())
        #         list_measurement_values.append(value)
        #     # categories = lines[-1].strip().split(",")
        #     categories = ["empty"] * len(list_measurement_values)
        #     rock_lengths = [abs(list_measurement_values[i+1] - list_measurement_values[i]) * 100 for i in range(len(list_measurement_values)-1)]
        #     return rock_lengths, categories

        # @st.dialog("Input measurement")
        # def input_measurement(row):
        #     measure_data = st.text_area(f"Measurement row {row}", key=f"measurement_{row}")
        #     submit = st.button("Submit", key=f"submit_{row}")
        #     if submit:
        #         data = load_measurement_data(measure_data)
        #         st.session_state.data[row] = data
        #         pre_saved_data[row] = data
        #         with open(f"{scan_folder}/{hole_id}/{box_from}_{box_to}/save_records.json", "w") as f:
        #             json.dump(pre_saved_data, f)
        #         st.rerun()

        # for i, btn in enumerate(measure_btns):
        #     if btn:
        #         input_measurement(i)

        if len(st.session_state.choossen) > 0:
            st.write(st.session_state.choossen)
            category = st.selectbox("Category", list_categories)
            change_btn = st.button("Change")
            clear_btn = st.button("Clear")
            if clear_btn:
                st.session_state.choossen = set()
                st.rerun()
            if change_btn:
                for choosen in st.session_state.choossen:
                    row, click = map(int, choosen.split("_"))
                    rock_lengths, categories, points_coord = st.session_state.data[row]
                    categories[click] = category
                    st.session_state.data[row] = [rock_lengths, categories, points_coord]
                #     pre_saved_data[row] = [rock_lengths, categories]
                
                # st.session_state.choossen = set()
                # with open(f"{scan_folder}/{hole_id}/{box_from}_{box_to}/save_records.json", "w") as f:
                #     json.dump(pre_saved_data, f)
                st.rerun()
    else:
        st.warning("Please enter valid box information!")