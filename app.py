import streamlit as st

st.set_page_config(page_title="EcoBici EDAV", page_icon=":bike:",initial_sidebar_state="expanded")

pages_data = [
         # first one is the default one
        {"file": r"pages/stations_map_page.py", "title":"Stations & Bike availability" ,"icon":":material/map_search:" },
        {"file": r"pages/route_page.py", "title":"Route finder" ,"icon":":material/route:" },
        {"file": r"pages/EDAV_page.py", "title":"Data analysis" ,"icon":":material/monitoring:" }
]

all_pages = [st.Page(page['file'],title=page['title'], icon=page['icon']) for page in pages_data]

pg = st.navigation(all_pages)
pg.run()