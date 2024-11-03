import streamlit as st

st.set_page_config(
    page_title="SC Markup",
    layout="wide",
)

pages = {
    "Main" : [
        st.Page("subpages/markup.py", title="Markup"),
        st.Page("subpages/config.py", title="Configurations"),
    ]
}

pg = st.navigation(pages)
pg.run()
