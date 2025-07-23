import streamlit as st

edav_page = st.Page("EDAV_page.py", title="Data analysis of historical data", icon=":material/monitoring:")
# delete_page = st.Page("delete.py", title="Delete entry", icon=":material/delete:")

pg = st.navigation([edav_page])
st.set_page_config(page_title="EcoBici EDAV", page_icon=":bike:")
pg.run()