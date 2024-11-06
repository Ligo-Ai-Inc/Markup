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
    ],
    "Data":[
        st.Page("subpages/rqd.py", title="RQD", icon=":material/stacked_bar_chart:"),
        st.Page("subpages/recovery.py", title="Recovery", icon=":material/stacked_bar_chart:"),
    ]
}

pg = st.navigation(pages)
pg.run()

