import streamlit as st  
import pandas as pd

df = pd.read_excel("assets/data.xlsx")

st.dataframe(df)