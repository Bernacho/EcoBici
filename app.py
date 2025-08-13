import streamlit as st

import os
os.environ["STREAMLIT_WATCHDOG_MODE"] = "poll"


st.set_page_config(page_title="EcoBici EDAV", page_icon=":bike:",initial_sidebar_state="expanded")

pages_data = [
         # first one is the default one
        {"file": r"pages/stations_map.py", "title":"Stations & Bike availability" ,"icon":":material/map_search:" },
        {"file": r"pages/route.py", "title":"Route finder" ,"icon":":material/route:" },
        {"file": r"pages/EDAV.py", "title":"Data analysis" ,"icon":":material/monitoring:" }
]

all_pages = [st.Page(page['file'],title=page['title'], icon=page['icon']) for page in pages_data]

pg = st.navigation(all_pages)
pg.run()