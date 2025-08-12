import streamlit as st
import pydeck as pdk
import pandas as pd
import requests
import json
import numpy as np
from matplotlib import colors
import osmnx as ox
from geopandas.tools import geocode
from math import radians, sin, cos, sqrt, atan2,asin,log2
import networkx as nx
import os
import logging

import tempfile



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ox.settings.timeout = 600  # seconds
ox.settings.overpass_settings = '[out:json][timeout:600]'


ecobici_colors = ['#009844','#B1B1B1','#235B4E','#483C47','#7D5C65','#FFFFFF','#FDE74C','#D81E5B']
WALKING_SPEED = 4.8 #km/hr
BIKE_SPEED = 20 #km/hr

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

@st.cache_data
def load_stations_data():
    api_urls = get_api_urls()
    response = requests.get(api_urls['station_information'],timeout=3.1)
    stations = pd.DataFrame.from_dict(response.json()['data']['stations'])
    stations['short_name'] = stations['short_name'].str.pad(width=3, side='left', fillchar='0')
    stations.set_index("station_id",inplace=True)
    
    return stations


GRAPH_PATH = "cached_graph.graphml"

@st.cache_resource
def load_graph():
    
    if os.path.exists(GRAPH_PATH):
            logger.info("Starting graph loading...")
            G = ox.load_graphml(GRAPH_PATH)
            logger.info("... Done")
    else:
        # logger.info("Starting graph download...")
        # stations_df = load_stations_data()
        # buffer = 0.07
        # north = stations_df.lat.max() + buffer
        # south = stations_df.lat.min() - buffer
        # east = stations_df.lon.max()+buffer
        # west = stations_df.lon.min()-buffer

        # logger.info(f"OX timeout set to {ox.settings.timeout} seconds")
        # G = ox.graph_from_bbox((west, south, east, north), network_type="bike")
        # G = ox.distance.add_edge_lengths(G)

        graph_raw_url = "https://github.com/Bernacho/EcoBici_Dataset/raw/refs/heads/main/graphs/mexico_city.graphml"
        logger.info("Starting graph download from Github...")
        response = requests.get(graph_raw_url)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(suffix=".graphml", delete=False) as tmp_file:
                tmp_file.write(response.content)
                tmp_path = tmp_file.name

        G=  ox.load_graphml(tmp_path)
        logger.info("... Done.. Saving ...")

        ox.save_graphml(G, GRAPH_PATH)
        logger.info("... Done")

    bounds = ox.convert.graph_to_gdfs(G, nodes=True, edges=False).total_bounds
    
    G_walk = nx.MultiGraph(G)

    return G, bounds, G_walk

@st.cache_data
def geocode_address(address, crs=4326):
    g_code = geocode(address, provider='nominatim', 
                user_agent="ecobici_app",timeout=10).to_crs(crs)
    
    return g_code

@st.cache_data
def get_api_urls():
    ECOBICI_API = "https://gbfs.mex.lyftbikes.com/gbfs/gbfs.json"
    response = requests.get(ECOBICI_API,timeout=2)
    api_urls = {x['name']: x['url'] for x in response.json()['data']['en']['feeds']}

    return api_urls

def get_stations_status():
    api_urls = get_api_urls()
    status_response = requests.get(api_urls['station_status'])
    status = pd.DataFrame.from_dict(status_response.json()['data']['stations'])
    status.set_index("station_id",inplace=True)
    
    return status


def compute_zoom(min_lat, max_lat, min_lon, max_lon, padding=1.0):
    lat_span = max_lat - min_lat
    lon_span = max_lon - min_lon

    zoom_lon = log2(360 / lon_span) if lon_span > 0 else 20
    zoom_lat = log2(180 / lat_span) if lat_span > 0 else 20

    zoom = min(zoom_lon, zoom_lat) - padding
    return max(zoom, 1) 

def get_map(cycling_route,walking_route_one,walking_route_two,address,availability):
    station_palette = {"green":ecobici_colors[0],
                       "yellow":ecobici_colors[-2],
                       "red":ecobici_colors[-1],
                       'white':ecobici_colors[-3]}
    station_palette = {k:colors.to_rgb(i) for k,i in station_palette.items()}
    avg_lon = set([x[0] for x in cycling_route]+ [x[0] for x in walking_route_one]+[x[0] for x in walking_route_two])
    avg_lon =round( sum(avg_lon)/len(avg_lon),4)
    avg_lat = set([x[1] for x in cycling_route]+ [x[1] for x in walking_route_one]+[x[1] for x in walking_route_two])
    avg_lat = round(sum(avg_lat)/len(avg_lat),4)

    all_routes = walking_route_one+walking_route_two+cycling_route
    zoom_level = compute_zoom(min([x[1] for x in all_routes]),max([x[1] for x in all_routes]),
                              min([x[0] for x in all_routes]),max([x[0] for x in all_routes]),0)
    
    view_state = pdk.ViewState(
        latitude=avg_lat,
        longitude=avg_lon,
        zoom=zoom_level,
        pitch= 0
    )

    scatter_dots = [
        ["origin",walking_route_one[0][0],walking_route_one[0][1],'red',f"Origin: {address.iloc[0]}"],
        ["origin_station",cycling_route[0][0],cycling_route[0][1],'green',
         f"Station: {availability[0][0]}<br/>Bikes available: {availability[0][1]}"],
        ["destination_station",cycling_route[-1][0],cycling_route[-1][1],'green',
         f"Station: {availability[1][0]}<br/>Docks available: {availability[1][1]}"],
        ["destination",walking_route_two[-1][0],walking_route_two[-1][1],'red',f"Destination: {address.iloc[1]}"],
    ]
    scatter_df = pd.DataFrame(scatter_dots,columns=['node','lon','lat','color',"tooltip_text"])
    scatter_df['edge_color'] = scatter_df.color.apply(lambda x: tuple(int(y*255) for y in station_palette[x]))
    scatter_df['fill_color'] = scatter_df.apply(lambda x: "white" if x.node=="destination" else x.color ,axis=1)
    scatter_df['fill_color'] = scatter_df.fill_color.apply(lambda x: tuple(int(y*255) for y in station_palette[x]))
    

    layer = pdk.Layer(
            "ScatterplotLayer",
            data=scatter_df,
            stroked=True,
            filled=True,
            get_position=["lon", "lat"],
            get_fill_color='fill_color',
            get_line_color="edge_color",
            get_radius=30,
            line_width_min_pixels=3,
            pickable=True
        )
    
    path_df = pd.DataFrame({"name":["w1","bike","w2"],"path": [walking_route_one,cycling_route,walking_route_two]}) 
    path_df['c'] = path_df['name'].apply(lambda x: 'red' if x[:1]=="w" else 'green')
    path_df['color'] = path_df.c.apply(lambda x: tuple(int(y*255) for y in station_palette[x]))
    path_layer1 = pdk.Layer(
        "PathLayer",
        data=path_df,
        get_path="path",
        get_color='color',
        get_width=5,
        width_min_pixels=5,
        pickable=False
    )

    
    tooltip = {
        "html": """
            <div style="max-width: 200px; word-wrap: break-word;">
                <b>{tooltip_text}</b>
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


    deck = pdk.Deck(
        initial_view_state=view_state,
        layers=[path_layer1,layer],
        map_style='dark',
        tooltip=tooltip
    )
    
    return deck

def coordinate_in_bounds(G_bounds,coordinate):
    lon,lat = coordinate
    west, south, east, north = G_bounds
    is_inside = (south <= lat <= north) and (west <= lon <= east)

    return is_inside

def haversine(point1, point2):
    lon1, lat1 = point1
    lon2, lat2 = point2
    R = 6371.0  
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

def find_closest_node_by_path(G, source_node,destination_node, target_nodes):
    origin_point = np.array((G.nodes[source_node]['x'], G.nodes[source_node]['y']))
    direction_point = np.array((G.nodes[destination_node]['x'], G.nodes[destination_node]['y']))
    
    direction_vector = direction_point - origin_point
    direction_vector = direction_vector / np.linalg.norm(direction_vector)  # Normalize
    
    # Check neighbors
    best_node = None
    best_cost = float('inf')


    for direction_threshold in [0.7,0.5,0.3,0,-1]:
        if best_node is not None:
            break

        for node in target_nodes:
            if node == source_node:
                continue
            node_point = np.array((G.nodes[node]['x'], G.nodes[node]['y']))
            node_vector = node_point - origin_point
            if np.linalg.norm(node_vector) == 0:
                continue
            node_vector = node_vector / np.linalg.norm(node_vector) # normalize
            similarity = np.dot(direction_vector, node_vector)

            if similarity >= direction_threshold:
                try:
                    path_length = nx.shortest_path_length(G, source_node, node, weight='length')
                    if path_length < best_cost:
                        best_cost = path_length
                        best_node = node
                except nx.NetworkXNoPath:
                    continue

    return best_node




def get_closest_stations(G,stations,node,type,how_many=5):
    ref_point = (G.nodes[node]['x'],G.nodes[node]['y'])
    stations['distance_km'] = stations.apply(lambda row: haversine(ref_point, (row['lon'], row['lat'])), axis=1)
    if type=="origin":
        stations['availability'] = stations['num_bikes_available']
    else:
        stations['availability'] = stations['num_docks_available']
    return stations[stations['availability']>0].sort_values("distance_km",ascending=True).iloc[:how_many]

def get_closest_station(G,stations,node,destination_node,type):
    closest_stations = get_closest_stations(G,stations,node,type)
    if len(closest_stations)==0:
        return None
    closest_stations['node'] = closest_stations.apply(lambda x: ox.distance.nearest_nodes(G, x.lon,x.lat),axis=1)

    closest_ = find_closest_node_by_path(G,node,destination_node, closest_stations.node.tolist())
    return closest_stations[closest_stations.node==closest_].iloc[0]


def main():
    st.set_page_config(layout="wide", page_title="Route finder", page_icon=":bike:")
    st.markdown("""
        <style>
            .block-container {
                padding-top: 1rem;
            }
        </style>
    """, unsafe_allow_html=True)

    if "selected" not in st.session_state:
        st.session_state.selected = "rent" 
    
    st.title("🚲 Ecobici - Mexico City Bike Sharing System") 
    st.header("Route finder") 
    
    with st.spinner("Loading graph... This may take up to a minute."):
        G, G_bounds, G_walk = load_graph()

    
    st.write("Enter addresses for the *origin* and *destination* of your tip.")
    st.write("We will let you know which are the closest stations and the best route to follow.")

    with st.form("ori_des_form"):
        col1,col2 = st.columns(2)
        with col1:
            origin = st.text_input("Origin address")

        with col2:
            destination = st.text_input("Destination address")
        submitted = st.form_submit_button("Submit")

    st.write("---")
    if submitted==False:
        st.info("Input an origin and destination to get your route")
    else:
        if origin=="":
            st.warning("Input an origin")
            st.stop()
        if destination=="":
            st.warning("Input a destination")
            st.stop()

        if (origin!="") and (destination!=""):
            addresses = geocode_address([origin,destination])
            
            try:
                origin_coordinate = (addresses.iloc[0].geometry.x,addresses.iloc[0].geometry.y)
            except:
                origin_coordinate = None
            try:
                destination_coordinate = (addresses.iloc[1].geometry.x,addresses.iloc[1].geometry.y)
            except:
                destination_coordinate = None
            
            if origin_coordinate is None:
                st.warning("Origin address not found!")
                st.stop()
            if destination_coordinate is None:
                st.warning("Destination address not found!")
                st.stop()

            if coordinate_in_bounds(G_bounds,origin_coordinate)==False:
                origin_coordinate = None
                st.warning("Origin address too far from the bikes system!")
                st.stop()

            if coordinate_in_bounds(G_bounds,destination_coordinate)==False:
                destination_coordinate = None
                st.warning("Destination address too far from the bikes system!")
                st.stop()

            if (origin_coordinate is not None) and (destination_coordinate is not None):
                origin_node = ox.nearest_nodes(G,origin_coordinate[0],origin_coordinate[1])
                destination_node = ox.nearest_nodes(G,destination_coordinate[0], destination_coordinate[1])

                stations = load_stations_data()
                station_status = get_stations_status()
                stations = stations.join(station_status[['num_bikes_available','num_docks_available']])

                origin_station = get_closest_station(G_walk,stations,origin_node,destination_node,"origin")
                dis_to_ori_station = 0 if origin_station is None else nx.shortest_path_length(G_walk,origin_node,origin_station['node'],weight='length')
                if dis_to_ori_station>=5000:
                    st.warning("The origin address is too far to a station!")
                    origin_station= None

                destination_station = get_closest_station(G_walk,stations,destination_node,origin_node,"destination")
                dis_from_des_station = 0 if destination_station is None else nx.shortest_path_length(G_walk,destination_station['node'],destination_node,weight='length')
                if dis_from_des_station>=5000:
                    st.warning("The destination address is too far to a station!")
                    destination_station= None

                if (origin_station is None) or (destination_station is None):
                    st.error("We could not find a station for your origin or your destination address!")
                    st.stop()
                else:
                    route_to_ori_station = nx.shortest_path(G_walk, origin_node, origin_station['node'], weight='length')
                    route_from_des_station = nx.shortest_path(G_walk, destination_station['node'],destination_node, weight='length')

                    route_bike = nx.shortest_path(G,  origin_station['node'],destination_station['node'], weight='length')
                    dis_bike= nx.shortest_path_length(G,origin_station['node'],destination_station['node'],weight='length')

                    est_time_hrs = (((dis_to_ori_station+dis_from_des_station)/1000)/WALKING_SPEED) + ((dis_bike/1000)/BIKE_SPEED)
                    est_time_min = np.ceil(est_time_hrs*60)

                    st.success("Best route found!")

                    col3,col4 = st.columns(2)
                    
                    bikes_available = station_status.loc[origin_station.name,"num_bikes_available"]
                    docks_available = station_status.loc[destination_station.name,"num_docks_available"]
                    with col3:
                        st.write(f"Origin station: {origin_station['name']}")
                        st.write(f"Available bikes: {bikes_available}")

                    with col4:
                        st.write(f"Destination station: {destination_station['name']}")
                        st.write(f"Available docks: {docks_available}")

                    cola, colb, colc = st.columns(3)
                    cola.metric("🚴 Riding Distance", f"{dis_bike/1000:,.1f} km")
                    colb.metric("🚶 Walking Distance", f"{(dis_to_ori_station+dis_from_des_station)/1000:,.1f} km")
                    colc.metric("⏱️ Estimated Time", f"{est_time_min:,.0f} min")

                    walking_route_one  = [[G.nodes[node]['x'], G.nodes[node]['y']]
                                           for node in route_to_ori_station]
                    walking_route_two  = [[G.nodes[node]['x'], G.nodes[node]['y']]
                                           for node in route_from_des_station]
                    cycling_route  = [[G.nodes[node]['x'], G.nodes[node]['y']]
                                       for node in route_bike]

                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; width: 100%; padding: 10px 0;">
                    <div style="display: flex; align-items: center;">
                        <div style="width: 16px; height: 16px; border-radius: 50%; background-color: {ecobici_colors[-1]}; border: 2px solid {ecobici_colors[-1]}; margin-right: 6px;"></div>
                        <span>Origin</span>
                    </div>
                    <div style="display: flex; align-items: center;">
                        <div style="width: 16px; height: 16px; border-radius: 50%; background-color: {ecobici_colors[0]}; border: 2px solid {ecobici_colors[0]}; margin-right: 6px;"></div>
                        <span>Bicycle stations</span>
                    </div>
                    <div style="display: flex; align-items: center;">
                        <div style="width: 16px; height: 16px; border-radius: 50%; background-color: {ecobici_colors[-3]}; border: 2px solid {ecobici_colors[-1]}; margin-right: 6px;"></div>
                        <span>Destination</span>
                    </div>
                    <div style="display: flex; align-items: center;">
                        <div style="width: 30px; height: 0; border-top: 4px solid {ecobici_colors[0]}; margin-right: 6px;"></div>
                        <span>Bicycle Path</span>
                    </div>
                    <div style="display: flex; align-items: center;">
                        <div style="width: 30px; height: 0; border-top: 4px solid {ecobici_colors[-1]}; margin-right: 6px;"></div>
                        <span>Walking Path</span>
                    </div>
                    </div>
                    """, unsafe_allow_html=True)


                    deck = get_map(cycling_route,walking_route_one,walking_route_two,addresses.address,
                                   [[origin_station['name'],bikes_available],
                                    [destination_station['name'],docks_available]])
                    st.pydeck_chart(deck)

    
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


    st.write("---")
    st.write("#### Data Source: [Mexico City Ecobici](https://www.ecobici.cdmx.gob.mx/)")

if __name__ == "__main__":
    main()