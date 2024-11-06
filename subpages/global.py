import streamlit as st

nrow = st.number_input("Number of rows", min_value=1, max_value=5, value=5)
st.session_state.nrow = nrow
box_length = st.number_input("Box length", min_value=0, max_value=1000, value=60) # cm
st.session_state.box_length = box_length