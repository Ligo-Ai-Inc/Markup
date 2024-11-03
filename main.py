import streamlit as st
from PIL import Image
from st_clickable_images import clickable_images
import base64
from io import BytesIO
import json
import os

measure_btns = []

if "data" not in st.session_state:
    st.session_state.data = {}
if "pre_loaded" not in st.session_state:
    st.session_state.pre_loaded = False

list_categories = ["rock <10cm", "rock >10cm", "crumbly", "block"]
color_map = {
    "rock <10cm": "#454b54",
    "rock >10cm": "#1DA0A5",
    "crumbly": "#f2c335",
    "block": "#f28a35"
}

box_length = st.number_input("Box length", min_value=0, max_value=1000, value=60) # cm
col1, col2 = st.columns(2)
box_from = col1.number_input("Box from", min_value=0.0, max_value=1000.0, value=0.000)
box_to = col2.number_input("Box to", min_value=0.0, max_value=1000.0, value=0.000)
nrow = st.number_input("Number of rows", min_value=1, max_value=5, value=5)

pre_saved_data = {}
if os.path.exists("save_records.json"):
    with open("save_records.json", "r") as f:
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

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        html = f"data:image/png;base64,{img_str}"
        list_images.append(html)

    return list_images

list_clicks = []
for i in range(nrow):
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
                div_style={"display": "flex", "justify-content": "space-between", "background-color": "white", "width": "100%"},
                img_style={"margin": "5px", "height": "25px"},
            )
            list_clicks.append(clicked)

for row, click in enumerate(list_clicks):
    if click != -1:
        rock_lengths, categories = st.session_state.data[row]
        st.write(f"Row {row+1} rock {click+1}")
        category_index = list_categories.index(categories[click])
        st.selectbox("Category", list_categories, key=f"category_{row}_{click}", index=category_index)

change_btn = st.button("Change")
if change_btn:
    for row, (rock_lengths, categories) in st.session_state.data.items():
        for i, category in enumerate(categories):
            if f"category_{row}_{i}" in st.session_state:
                categories[i] = st.session_state[f"category_{row}_{i}"]
        st.session_state.data[row] = [rock_lengths, categories]
        pre_saved_data[row] = [rock_lengths, categories]
        
    with open("save_records.json", "w") as f:
        json.dump(pre_saved_data, f)
    st.rerun()

def load_measurement_data(measure_data):
    lines = measure_data.split("\n")
    list_measurement_values = []
    for line in lines[:-1]:
        value = float(line.split(":")[-1].strip())
        list_measurement_values.append(value)
    categories = lines[-1].strip().split(",")
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
        with open("save_records.json", "w") as f:
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
