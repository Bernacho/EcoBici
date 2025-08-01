import streamlit as st
import pydeck as pdk
import pandas as pd
import requests
import json
import numpy as np
from matplotlib import colors

ecobici_colors = ['#009844','#B1B1B1','#235B4E','#483C47','#7D5C65','#FFFFFF','#FDE74C','#D81E5B']

@st.cache_data
def get_api_urls():
    ECOBICI_API = "https://gbfs.mex.lyftbikes.com/gbfs/gbfs.json"
    response = requests.get(ECOBICI_API,timeout=2)
    api_urls = {x['name']: x['url'] for x in response.json()['data']['en']['feeds']}

    return api_urls

@st.cache_data(ttl=60)
def load_stations_data():
    api_urls = get_api_urls()
    response = requests.get(api_urls['station_information'],timeout=3.1)
    stations = pd.DataFrame.from_dict(response.json()['data']['stations'])
    stations['short_name'] = stations['short_name'].str.pad(width=3, side='left', fillchar='0')
    stations.set_index("station_id",inplace=True)
    
    return stations

@st.cache_data(ttl=60)
def get_stations_status():
    api_urls = get_api_urls()
    status_response = requests.get(api_urls['station_status'])
    status = pd.DataFrame.from_dict(status_response.json()['data']['stations'])
    status.set_index("station_id",inplace=True)
    
    return status

def get_map(on_3d):
    station_palette = {"green":ecobici_colors[0],
                       "yellow":ecobici_colors[-2],
                       "red":ecobici_colors[-1]}
    station_palette = {k:colors.to_rgb(i) for k,i in station_palette.items()}
   
    stations = load_stations_data()
    status = get_stations_status()
    stations = stations.join(status,rsuffix="_status").assign(
        num_available =lambda x: x.num_bikes_available if st.session_state.selected=="rent" else x.num_docks_available
    ).assign(
        fill_color= lambda x: pd.cut(x.num_available,bins=[0,1,3,np.inf],labels=['red','yellow','green'],right=False)
    ).assign(
        fill_color=lambda x: x.fill_color.astype(str).map(station_palette)
    ).assign(
        r=lambda x: x.fill_color.str[0]*255
    ).assign(
        g=lambda x: x.fill_color.str[1]*255
    ).assign(
        b=lambda x: x.fill_color.str[2]*255
    ).assign(name_code=lambda x: x.name.apply(lambda y: y.split(" ")[0])).assign(
        name_text =lambda x: x.name.apply(lambda y: " ".join(y.split(" ")[1:]))
    )

    view_state = pdk.ViewState(
        latitude=19.4326,
        longitude=-99.1332,
        zoom=12,
        pitch=45 if on_3d else 0
    )

    if st.session_state.selected=="rent":
        html_availability = """
            <b>Bikes available:</b> {num_bikes_available}<br/>
            <b>Capacity:</b> {capacity}<br/>
            <b>Bikes disabled:</b> {num_bikes_disabled}<br/>
            """
    else:
        html_availability = """
            <b>Docks available:</b> {num_docks_available}<br/>
            <b>Capacity:</b> {capacity}<br/>
            <b>Docks disabled:</b> {num_docks_disabled}<br/>
            """
    
    tooltip = {
        "html": """
            <div style='text-align: left'>
                <div style='font-weight: bold; font-size: 14px; margin-bottom: 4px'>
                    Station {name_code}<br/>
                    {name_text}
                </div>
                <hr style='border: none; border-top: 1px solid """+ecobici_colors[2]+"""; margin: 4px 0;' />
                <div style='font-size: 12px'>
                    """+html_availability+"""
                    <b>Electric:</b> {is_charging}
                </div>
            </div>
        """,
        "style": {
            "backgroundColor": f"{ecobici_colors[1]}",
            "color": f"{ecobici_colors[3]}",
            "fontSize": "12px",
            "padding": "5px",
            "borderRadius": "5px",

        }
    }

    if on_3d:
        layer = pdk.Layer(
            "ColumnLayer",
            data=stations,
            get_position=["lon", "lat"],
            get_elevation="num_available",
            elevation_scale=100 if view_state.zoom > 13 else 50,
            radius=40,
            get_fill_color="""
                [
                    num_available >= 3 ? 0 : (num_available>0 ? 253 : 216),
                    num_available >= 3 ? 152 : (num_available>0 ? 231 : 30),
                    num_available >= 3 ? 68 : (num_available>0 ? 76 : 91),
                    122
                ]
            """,
            pickable=True,
            extruded=True,
        )
    else:
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=stations,
            get_position=["lon", "lat"],
            get_fill_color=["r", "g", "b"],
            get_radius=40,
            pickable=True
        )

    deck = pdk.Deck(
        initial_view_state=view_state,
        layers=[ layer],
        map_style='dark', 
        tooltip=tooltip
    )

    return deck

def main():
    st.set_page_config(layout="wide", page_title="Station status", page_icon=":bike:")
    st.markdown("""
        <style>
            .block-container {
                padding-top: 1rem;
            }
        </style>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("""
        <style>
        .tooltip-label {
            font-size: 0.75rem;
        }
        </style>

        <span class="tooltip-label bottom-span" title="
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

    if "selected" not in st.session_state:
        st.session_state.selected = "rent" 
    
    st.title("🚲 Ecobici - Mexico City Bike Sharing System") 
    st.header("Station status")
    st.write("Query live data to search for available stations!")

    def select_a():
        st.session_state.selected = "rent"

    def select_b():
        st.session_state.selected = "return"

    col3, col4 = st.columns(2)
    with col3:
        st.button("Rent bike",icon=":material/pedal_bike:", on_click=select_a,type="primary" if st.session_state.selected == "rent" else "secondary")

    with col4:
        st.button("Return bike",icon=":material/bike_dock:", on_click=select_b,type="primary" if st.session_state.selected == "return" else "secondary")

    on_3d = st.toggle("3D view")
    deck = get_map(on_3d)
    st.pydeck_chart(deck)

    st.write("---")
    st.write("#### Data Source: [Mexico City Ecobici](https://www.ecobici.cdmx.gob.mx/)")

if __name__ == "__main__":
    main()