import streamlit as st
from PIL import Image, ImageDraw
from st_clickable_images import clickable_images
import base64
from io import BytesIO
import json
import os

measure_btns = []
scan_folder = "scans"
os.makedirs(scan_folder, exist_ok=True)

if "data" not in st.session_state:
    st.session_state.data = {}
if "pre_loaded" not in st.session_state:
    st.session_state.pre_loaded = False
if "nrow" not in st.session_state:
    st.session_state.nrow = 5
if "box_length" not in st.session_state:
    st.session_state.box_length = 60
if "prev_click" not in st.session_state:
    st.session_state.prev_click = None
if "prev_row_click" not in st.session_state:
    st.session_state.prev_row_click = [-1] * st.session_state.nrow
if "choossen" not in st.session_state:
    st.session_state.choossen = set()

list_categories = ["empty", "rock <10cm", "rock >10cm", "crumbly", "block"]
color_map = {
    "empty": "#ff6361",
    "rock <10cm": "#58508d",
    "rock >10cm": "#003f5c",
    "crumbly": "#bc5090",
    "block": "#ffa600"
}

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

# box_length = st.number_input("Box length", min_value=0, max_value=1000, value=60) # cm
box_length = st.session_state.box_length
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
        st.session_state.box_from = box_from
        st.session_state.box_to = box_to

        os.makedirs(f"{scan_folder}/{hole_id}/{box_from}_{box_to}", exist_ok=True)
        pre_saved_data = {}
        if os.path.exists(f"{scan_folder}/{hole_id}/{box_from}_{box_to}/save_records.json"):
            with open(f"{scan_folder}/{hole_id}/{box_from}_{box_to}/save_records.json", "r") as f:
                try:
                    pre_saved_data = json.load(f)
                except:
                    pass


        def gen_images(rock_lengths, categories):
            full_width = 1000
            height = 50
            list_images = []
            for rock_length, category in zip(rock_lengths, categories):
                rock_percentage = (rock_length / box_length)
                width = int(full_width * rock_percentage)
                image = Image.new("RGB", (width, height), color_map[category])
                draw = ImageDraw.Draw(image)
                draw.line((0, 0, 0, height), fill="black", width=5)
                draw.line((width-2, 0, width-2, height), fill="black", width=5)

                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                html = f"data:image/png;base64,{img_str}"
                list_images.append(html)

            return list_images

        save_btn = st.button("Save")

        list_clicks = []
        for i in range(st.session_state.nrow):
            col1, col2 = st.columns([0.2, 0.8])
            btn = col1.button("Measure row {}".format(i+1))   
            measure_btns.append(btn)

            if i in st.session_state.data:
                rock_lengths, categories = st.session_state.data[i]
                list_images = gen_images(rock_lengths, categories)

                with col2.container(border=False):
                    clicked = clickable_images(
                        list_images,
                        titles=[f"{i}_{j}" for j in range(len(list_images))],
                        div_style={"display": "flex", "justify-content": "space-between", "background-color": "white", "gap": "0px"},
                        img_style={"margin": "0px", "height": "20px"},
                    )
                    list_clicks.append(clicked)

            if "row_images" in st.session_state:
                col2.image(st.session_state.row_images[i], use_column_width=True)

        for row, click in enumerate(list_clicks):
            st.write(f"Row {row+1}: {click}", st.session_state.prev_row_click[row])
            if click != st.session_state.prev_row_click[row]:
                st.session_state.prev_row_click[row] = click
                st.session_state.choossen.add(f"{row}_{click}")

            # if click != -1 and f"{row}_{click}" != st.session_state.prev_click:
                # st.session_state.prev_click = f"{row}_{click}"
                # st.session_state.choossen.add(f"{row}_{click}")

        def load_measurement_data(measure_data):
            lines = measure_data.split("\n")
            list_measurement_values = []
            for line in lines[:]:
                value = float(line.split(":")[-1].strip())
                list_measurement_values.append(value)
            # categories = lines[-1].strip().split(",")
            categories = ["empty"] * len(list_measurement_values)
            rock_lengths = [abs(list_measurement_values[i+1] - list_measurement_values[i]) * 100 for i in range(len(list_measurement_values)-1)]
            return rock_lengths, categories

        @st.dialog("Input measurement")
        def input_measurement(row):
            measure_data = st.text_area(f"Measurement row {row}", key=f"measurement_{row}")
            submit = st.button("Submit", key=f"submit_{row}")
            if submit:
                data = load_measurement_data(measure_data)
                st.session_state.data[row] = data
                pre_saved_data[row] = data
                with open(f"{scan_folder}/{hole_id}/{box_from}_{box_to}/save_records.json", "w") as f:
                    json.dump(pre_saved_data, f)
                st.rerun()

        if not st.session_state.pre_loaded:
            st.session_state.pre_loaded = True
            for k, v in pre_saved_data.items():
                st.session_state.data[int(k)] = v
            st.rerun()

        for i, btn in enumerate(measure_btns):
            if btn:
                input_measurement(i)

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
                    rock_lengths, categories = st.session_state.data[row]
                    categories[click] = category
                    st.session_state.data[row] = [rock_lengths, categories]
                    pre_saved_data[row] = [rock_lengths, categories]
                
                st.session_state.choossen = set()
                with open(f"{scan_folder}/{hole_id}/{box_from}_{box_to}/save_records.json", "w") as f:
                    json.dump(pre_saved_data, f)
                st.rerun()
    else:
        st.warning("Please enter valid box information!")