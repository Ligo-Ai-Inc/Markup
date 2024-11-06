import streamlit as st

st.set_page_config(
    page_title="SC Markup",
    layout="wide",
)

pages = {
    "Scan" : [
        st.Page("subpages/config.py", title="Photo", icon=":material/add_a_photo:"),
        st.Page("subpages/markup.py", title="Measure", icon=":material/full_stacked_bar_chart:"),
        st.Page("subpages/review.py", title="Review", icon=":material/gallery_thumbnail:"),
    ],
    "Configuration" : [
        st.Page("subpages/global.py", title="Global", icon=":material/settings:"),
    ]
}

pg = st.navigation(pages)
pg.run()

