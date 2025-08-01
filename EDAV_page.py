import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import json
import matplotlib.font_manager as fm
import matplotlib as mpl
import requests
from datetime import datetime
import pytz

import gcsfs


plt.style.use('seaborn-v0_8-darkgrid')
font_path = "./static/Inter_18pt-Regular.ttf"
inter_font = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)
bold_path = "./static/Inter_18pt-Bold.ttf"
bold_font = fm.FontProperties(fname=bold_path)
fm.fontManager.addfont(bold_path)
mpl.rcParams['font.family'] = inter_font.get_name()

plt.rc('font', size=14)             
plt.rc('axes', titlesize=16)        
plt.rc('axes', labelsize=14)        
plt.rc('xtick', labelsize=12)     
plt.rc('ytick', labelsize=12)       
plt.rc('legend', fontsize=12)       
plt.rc('figure', titlesize=18)

ecobici_colors = ['#009844','#B1B1B1','#235B4E','#483C47','#7D5C65','#FFFFFF','#D81E5B']
palette = sns.blend_palette(ecobici_colors,n_colors= len(ecobici_colors))
text_color = palette[3]
plt.rcParams.update({"text.color":text_color,
                     "axes.labelcolor":text_color,
                     "ytick.labelcolor":text_color,
                     "xtick.labelcolor":text_color,
                     "axes.edgecolor":text_color})


BASE_GCS_DATA_URL = "gs://bernacho-ecobici-datahub/partitioned_historical_data/"
GITHUB_REPO = "Bernacho/EcoBici"


@st.cache_data
def load_data(base_path, filters, _fs): 
    try:
        df = pd.read_parquet(base_path[5:], filters=filters, filesystem=_fs,engine="pyarrow")

        df['Ciclo_Estacion_Retiro'] = df['Ciclo_Estacion_Retiro'].str.pad(width=3, side='left', fillchar='0')
        df['Ciclo_Estacion_Arribo'] = df['Ciclo_Estacion_Arribo'].str.pad(width=3, side='left', fillchar='0')
        
        df['Fecha_Arribo'] = pd.to_datetime(df['Fecha_Arribo'], errors='coerce')
        df['Fecha_Retiro'] = pd.to_datetime(df['Fecha_Retiro'], errors='coerce')
        df['date_start'] = pd.to_datetime(df['date_start'], errors='coerce')
        df['date_end'] = pd.to_datetime(df['date_end'], errors='coerce')

        df.dropna(subset=['date_end', 'date_start','Fecha_Arribo','Fecha_Retiro'], inplace=True)

        df['trip_duration_minutes'] = df['duration'] / 60
        duration_bins = list(np.arange(0,121,5))+[np.inf]
        duration_labels = [f"[{i}-{i+5})" for i in duration_bins[:-2] ] + ["120+"]
        df['trip_duration_min_bucket'] = pd.cut(df['trip_duration_minutes'],bins=duration_bins,labels=duration_labels,right=False)
        df['trip_duration_min_bucket'] = pd.Categorical(df['trip_duration_min_bucket'], categories=duration_labels, ordered=True)

        df['start_day_of_week'] = df['date_start'].dt.day_name()
        df['end_day_of_week'] = df['date_end'].dt.day_name()


        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['start_day_of_week'] = pd.Categorical(df['start_day_of_week'], categories=day_order, ordered=True)
        df['end_day_of_week'] = pd.Categorical(df['end_day_of_week'], categories=day_order, ordered=True)

        df['year'] = df.year.astype(int)
        df['month'] = df.month.astype(int)
        df['station_system'] = "Old"
        df['station_system'] = df.station_system.mask((df.year>2022) | ((df.year==2022) & (df.month>=10)), "New")
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data from GCS: {e}.")
        return pd.DataFrame() 
    
@st.cache_data
def load_stations_data():
    ECOBICI_API = "https://gbfs.mex.lyftbikes.com/gbfs/gbfs.json"
    response = requests.get(ECOBICI_API,timeout=2)
    api_urls = {x['name']: x['url'] for x in response.json()['data']['en']['feeds']}
    response = requests.get(api_urls['station_information'],timeout=3.1)
    stations = pd.DataFrame.from_dict(response.json()['data']['stations'])
    stations['short_name'] = stations['short_name'].str.pad(width=3, side='left', fillchar='0')
    stations.set_index("short_name",inplace=True)

    old_stations = pd.read_csv("https://raw.githubusercontent.com/"+GITHUB_REPO+"/refs/heads/main/data/old_system_stations.csv",dtype={'short_name': str})
    old_stations['short_name'] = old_stations['short_name'].str.pad(width=3, side='left', fillchar='0')
    old_stations = old_stations.groupby("short_name").agg({"name":"first",'lat':'first','lon':'first','capacity':"sum"})
    old_stations.sort_values(["name","short_name"],inplace=True)
    old_stations['equal'] = old_stations.name==old_stations.name.shift()
    old_stations['n'] = old_stations.groupby("name").equal.cumsum()
    repeated_names= old_stations.groupby("name").equal.sum().loc[lambda x: x>0]
    old_stations['new_name'] = old_stations.name+ old_stations.apply(lambda x: "" if x['name'] not in repeated_names.index else "-"+str(x['n']+1),axis=1)
    old_stations.drop(columns=['equal','n','name'],inplace=True)
    old_stations.rename(columns={"new_name":"name"},inplace=True)

    stations = pd.concat([stations,old_stations[~old_stations.index.isin(stations.index)]],axis=0,ignore_index=False)
    stations['system'] = "Old"
    stations['system'] = stations['system'].mask(stations.external_id.notna(),"New")
    
    return stations

@st.fragment
def render_hourly_chart(filtered_data,station_names,stations):
    station_name = st.selectbox("Station",options=station_names,index=station_names.index(st.session_state.selected_station_name),key="dynamic_station_select")

    station_id = stations[stations.name==station_name].index[0] if station_name != "All" else None
    if station_name!="All":
        d1_hr = filtered_data[filtered_data['Ciclo_Estacion_Retiro']==station_id]
        d2_hr = filtered_data[filtered_data['Ciclo_Estacion_Arribo']==station_id]
    else:
        d1_hr = filtered_data
        d2_hr = filtered_data
    d1_hr = d1_hr.groupby(d1_hr.date_start.dt.hour).size().to_frame("Retiro")
    d2_hr = d2_hr.groupby(d2_hr.date_end.dt.hour).size().to_frame("Arribo")
    d_hr = d1_hr.join(d2_hr).fillna(0).reindex(list(np.arange(0,24)),fill_value=0)
    d_hr['difference'] =d_hr.Arribo - d_hr.Retiro
    d_hr['difference_abs'] = d_hr.difference.abs()
    d_plot_hour = d_hr[['difference']].sort_index().reset_index()
    d_plot_hour.columns = ['hour','difference']

    fig_imb_hour, ax_imb_hour = plt.subplots(figsize=(10, 5))
    ax_imb_hour.barh(d_plot_hour.hour,d_plot_hour.difference,color=[palette[0] if x>0 else palette[-1] for x in d_plot_hour.difference])
    ax_imb_hour.set_title('Ride Imbalance by Hour')
    ax_imb_hour.axvline(0,color=text_color,linestyle="--",lw=2)
    ax_imb_hour.set_xlabel('Trips Difference (Arrivals - Departures)')
    ax_imb_hour.set_ylabel('Hour')
    ax_imb_hour.xaxis.set_major_formatter(lambda x,p: f"{x:,.0f}")
    st.pyplot(fig_imb_hour)


def main():
    st.set_page_config(layout="wide", page_title="EcoBici EDAV", page_icon=":bike:")
    st.markdown("""
        <style>
            .block-container {
                padding-top: 1rem;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("🚲 Ecobici - Mexico City Bike Sharing System") 
    st.header("Exploratory Data Analysis")
    st.write("Select a year and optionally a month to load bike sharing trip data.")

    st.sidebar.header("Data Selection")
    tz = pytz.timezone("America/Mexico_City")
    now = datetime.now(tz)
    selected_year = st.sidebar.selectbox(
        "Select Year",
        options=list(range(2019, now.year)),
        index=None, 
        placeholder="Year..."
    )

    month_options = {
        "Month...": None,
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12
    }
    selected_month_name = st.sidebar.selectbox(
        "Select a Month (Optional)",
        options=list(month_options.keys()),
        index=0 
    )
    selected_month_num = month_options[selected_month_name]

    data = pd.DataFrame() 
    stations = load_stations_data()
    
    fs = None
    if "gcp_service_account" in st.secrets:
        gcp_sa_info = st.secrets["gcp_service_account"]
        if isinstance(gcp_sa_info, str):
            gcp_sa_info = json.loads(gcp_sa_info)
        
        gcp_sa_info_mutable = dict(gcp_sa_info) 
        if 'private_key' in gcp_sa_info_mutable and isinstance(gcp_sa_info_mutable['private_key'], str):
            gcp_sa_info_mutable['private_key'] = gcp_sa_info_mutable['private_key'].replace('\\n', '\n')

        fs = gcsfs.GCSFileSystem(token=gcp_sa_info_mutable)
    else:
        st.warning("No GCS service account found in Streamlit secrets. Attempting anonymous access.")
        fs = gcsfs.GCSFileSystem()


    if selected_year:
        filters = [('year', '=', selected_year)]
        if selected_month_num is not None:
            filters.append(('month', '=', selected_month_num))
        

        data = load_data(BASE_GCS_DATA_URL, filters, fs) 

        if selected_month_num is None:
            st.sidebar.warning(f"Month not selected. Loading all data for year {selected_year}.")
    else:
        st.info("Please select a year to load data.")


    if not data.empty:
        st.success("Data loaded successfully!")
        st.write("### Data Sample")
        st.dataframe(data.head())

        st.write("### Data Overview")
        st.write(f"Total trips: {len(data):,.0f}")
        min_date = data['date_end'].dt.date.min()
        max_date = data['date_end'].dt.date.max()
        st.write(f"Date range: {min_date} to {max_date}")
        st.write(f"Average trip duration: {data['trip_duration_minutes'].mean():,.2f} minutes")

        st.sidebar.header("Filters (for Loaded Data)")
        
        date_range = st.sidebar.date_input(
            "Select Date Range within Loaded Data",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        if len(date_range) == 2:
            filtered_data = data[(data['date_end'].dt.date >= date_range[0]) & (data['date_end'].dt.date <= date_range[1])]
        else:
            filtered_data = data 

        st.sidebar.write(f"Trips in selected range: {len(filtered_data):,.0f}")

        if filtered_data.empty:
            st.warning("No data available for the selected date range. Please adjust your filters.")
            return

        st.write("---")

  
        st.header("📊 Key Visualizations")

    
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Trips Over Time")
            trips_by_date = filtered_data.groupby(pd.Grouper(key="date_end",freq="D")).size().reset_index(name='trip_count')
            fig_time, ax_time = plt.subplots(figsize=(10, 5))
            ax_time.plot(trips_by_date['date_end'], trips_by_date['trip_count'], color=palette[-1],lw=3)
            ax_time.set_title(f'Daily Trip Counts for Selected Data')
            ax_time.set_xlabel('Date')
            ax_time.set_ylabel('Number of Trips')
            ax_time.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax_time.yaxis.set_major_formatter(lambda x,p: f"{x:,.0f}")
            fig_time.autofmt_xdate()
            st.pyplot(fig_time)

        with col2:
            st.subheader("Trips by Day of Week")
            trips_by_day = filtered_data['start_day_of_week'].value_counts().sort_index().reset_index(name='trip_count')
            trips_by_day.columns = ['day_of_week', 'trip_count'] 
            fig_day, ax_day = plt.subplots(figsize=(10, 5))
            sns.barplot(x='day_of_week', y='trip_count', data=trips_by_day, color=palette[0], ax=ax_day)
            ax_day.set_title('Trip Counts by Day of Week')
            ax_day.set_xlabel('Day of Week')
            ax_day.set_ylabel('Number of Trips')
            ax_day.yaxis.set_major_formatter(lambda x,p: f"{x:,.0f}")
            st.pyplot(fig_day)

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Trips by Hour of Day")
            trips_by_hour = filtered_data.assign(hour_of_day=lambda x: x.date_start.dt.hour).groupby('hour_of_day').size().sort_index().reset_index(name='trip_count')
            trips_by_hour.columns = ['hour_of_day', 'trip_count']
            fig_hour, ax_hour = plt.subplots(figsize=(10, 5))
            sns.barplot(x='hour_of_day', y='trip_count', data=trips_by_hour, color=palette[0], ax=ax_hour)
            ax_hour.set_title('Trip Counts by Hour of Day')
            ax_hour.set_xlabel('Hour of Day')
            ax_hour.set_ylabel('Number of Trips')
            ax_hour.yaxis.set_major_formatter(lambda x,p: f"{x:,.0f}")
            st.pyplot(fig_hour)

        with col4:
            st.subheader("Trip Duration Distribution")
            fig_duration, ax_duration = plt.subplots(figsize=(10, 5))
            trips_by_dur = filtered_data.groupby('trip_duration_min_bucket',observed=False).size().reset_index(name='trip_count')
            trips_by_dur.columns=['duration','trip_count']
            sns.barplot(x='duration',y='trip_count',data=trips_by_dur, color=palette[2], ax=ax_duration)
            ax_duration.set_title('Distribution of Trip Durations (minutes)')
            ax_duration.set_xlabel('Trip Duration (minutes)')
            ax_duration.set_ylabel('Frequency')
            ax_duration.set_xticks(np.arange(-0.5,24.5,1))
            ax_duration.set_xticklabels(list(np.arange(0,120,5))+["120+"])
            ax_duration.yaxis.set_major_formatter(lambda x,p: f"{x:,.0f}")
            st.pyplot(fig_duration)

        col5, col6 = st.columns(2)
        
        with col5:
            st.subheader("Top 10 Origin Stations")
            top_start_stations = filtered_data['Ciclo_Estacion_Retiro'].value_counts().head(10).sort_values().to_frame(name='trip_count')
            top_start_stations['station_name'] = top_start_stations.index.map(stations.name)
            fig_start_station, ax_start_station = plt.subplots(figsize=(7, 10))
            y = np.arange(top_start_stations.shape[0])
            ax_start_station.hlines(y=y,xmin=0,xmax=top_start_stations.trip_count,color=palette[0],linewidth=5)
            ax_start_station.scatter(top_start_stations.trip_count,y,color=palette[0],s=250)
            ax_start_station.set_yticks(y)
            ax_start_station.set_yticklabels(top_start_stations.station_name,fontsize=14)
            ax_start_station.set_xlim(left=0)
            ax_start_station.xaxis.set_major_formatter(lambda x,p: f"{x:,.0f}")
            ax_start_station.set_title('Top 10 Origin Stations by Trip Count')
            ax_start_station.set_xlabel('Number of Trips')
            ax_start_station.set_ylabel('Station')
            st.pyplot(fig_start_station)

        with col6:
            st.subheader("Top 10 Destination Stations")
            top_end_stations = filtered_data['Ciclo_Estacion_Arribo'].value_counts().head(10).sort_values().to_frame(name='trip_count')
            top_end_stations['station_name'] = top_end_stations.index.map(stations.name)
            fig_end_station, ax_end_station = plt.subplots(figsize=(7, 10))
            y = np.arange(top_end_stations.shape[0])
            ax_end_station.hlines(y=y,xmin=0,xmax=top_end_stations.trip_count,color=palette[2],linewidth=5)
            ax_end_station.scatter(top_end_stations.trip_count,y,color=palette[2],s=250)
            ax_end_station.set_yticks(y)
            ax_end_station.set_yticklabels(top_end_stations.station_name,fontsize=14)
            ax_end_station.set_xlim(left=0)
            ax_end_station.xaxis.set_major_formatter(lambda x,p: f"{x:,.0f}")
            ax_end_station.set_title('Top 10 Destination Stations by Trip Count')
            ax_end_station.set_xlabel('Number of Trips')
            ax_end_station.set_ylabel('Station')
            st.pyplot(fig_end_station)

        col7, col8 = st.columns(2)
        with col7:
            d1 = filtered_data['Ciclo_Estacion_Retiro'].value_counts()
            d2 = filtered_data['Ciclo_Estacion_Arribo'].value_counts()
            d = d1.to_frame("Retiro").join(d2.to_frame("Arribo")).fillna(0)
            d['difference'] =d.Arribo - d.Retiro
            d['difference_abs'] = d.difference.abs()
            d_plot = pd.concat([d.sort_values("difference").head(),d.sort_values("difference").tail()],axis=0)[['difference']].reset_index()
            d_plot.columns = ['station_id','difference']
            d_plot['station_name'] = d_plot.station_id.map(stations.name)

            st.subheader("Top 5 and Bottom 5 Stations with Most Imbalance")
            fig_imb, ax_imb = plt.subplots(figsize=(10, 10))
            sns.barplot(x='difference', y='station_name', data=d_plot, 
                        palette=[palette[0] if x>0 else palette[-1] for x in d_plot.difference],
                        hue='station_name',legend=False, ax=ax_imb)
            ax_imb.set_title('Ride Imbalance by Station')
            ax_imb.axvline(0,color=text_color,linestyle="--",lw=2)
            ax_imb.set_xlabel('Trips Difference (Arrivals - Departures)')
            ax_imb.set_ylabel('Station')
            ax_imb.xaxis.set_major_formatter(lambda x,p: f"{x:,.0f}")
            st.pyplot(fig_imb)

        if "selected_station_name" not in st.session_state:
            st.session_state.selected_station_name = "All"
        def update_station():
            st.session_state.selected_station_name = st.session_state.station_select

        with col8:
            st.subheader("Imbalance by Time of Day")
            st.write("Select the station to visualize its imbalance over the day.")
            all_stations_ids = set(filtered_data['Ciclo_Estacion_Retiro'].unique().tolist() + filtered_data['Ciclo_Estacion_Arribo'].tolist())
            station_names = ['All']+stations.reindex(all_stations_ids).name.sort_values().tolist()
            
            render_hourly_chart(filtered_data,station_names,stations)

        
        st.write("---")
        st.info("💡 **Tip:** Adjust the date range in the sidebar to explore specific periods!")
        st.write("#### Data Source: [Mexico City Ecobici](https://www.ecobici.cdmx.gob.mx/)")

    else:
        st.warning("No data loaded.")

if __name__ == "__main__":
    main()
