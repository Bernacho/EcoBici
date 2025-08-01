import streamlit as st

st.set_page_config(page_title="EcoBici EDAV", page_icon=":bike:",initial_sidebar_state="expanded")
st.config.set_option("client.showSidebarNavigation", False)

edav_page = st.Page("EDAV_page.py", title="Data analysis", icon=":material/monitoring:")
map_page = st.Page("stations_map_page.py", title="Stations & Bike availability", icon=":material/map_search:")
route_page = st.Page("route_page.py", title="Route finder", icon=":material/route:")

all_pages = [map_page,route_page,edav_page]

with st.sidebar:
        st.sidebar.markdown("""
        <style>
        .tooltip-label {
            font-size: 0.85rem;
            font-weight: 500;
        }
        </style>

        <span class="tooltip-label" title="
        • Python
        • OSMnx
        • NetworkX
        • Geopandas
        • NumPy
        • Streamlit
        • Matplotlib
        • Seaborn
        • Git & GitHub
        ">
        🛠️ <strong>Tools & Technologies Used</strong>
        </span>
        """, unsafe_allow_html=True)

        st.write("---")
        for page in all_pages:
                st.page_link(page.url_path+".py", label=page.title,icon=page.icon)

pg = st.navigation(all_pages)
pg.run()