import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import json # Import json to parse the service account key
import matplotlib.font_manager as fm
import matplotlib as mpl
import requests
import time
import uuid

# Ensure pyarrow, gcsfs, and fsspec are installed for Parquet and GCS support
# pip install pyarrow gcsfs fsspec

# Import gcsfs for explicit file system handling
import gcsfs

# Set Matplotlib style for better aesthetics
plt.style.use('seaborn-v0_8-darkgrid')
font_path = "./static/Inter_18pt-Regular.ttf"
inter_font = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)
bold_path = "./static/Inter_18pt-Bold.ttf"
bold_font = fm.FontProperties(fname=bold_path)
fm.fontManager.addfont(bold_path)
mpl.rcParams['font.family'] = inter_font.get_name()

plt.rc('font', size=14)              # Default text
plt.rc('axes', titlesize=16)         # Axes title
plt.rc('axes', labelsize=14)         # X/Y labels
plt.rc('xtick', labelsize=12)        # X tick labels
plt.rc('ytick', labelsize=12)        # Y tick labels
plt.rc('legend', fontsize=12)        # Legend
plt.rc('figure', titlesize=18)

ecobici_colors = ['#009844','#B1B1B1','#235B4E','#483C47','#7D5C65','#FFFFFF','#D81E5B']
palette = sns.blend_palette(ecobici_colors,n_colors= len(ecobici_colors))
text_color = palette[3]
plt.rcParams.update({"text.color":text_color,
                     "axes.labelcolor":text_color,
                     "ytick.labelcolor":text_color,
                     "xtick.labelcolor":text_color,
                     "axes.edgecolor":text_color})

# Base URL for the Google Cloud Storage bucket folder
# IMPORTANT: This path MUST start with 'gs://' and point to the root of your partitioned dataset.
# For your setup, this should be 'gs://bernacho-ecobici-datahub/ecobici_partitioned_data/'
BASE_GCS_DATA_URL = "gs://bernacho-ecobici-datahub/partitioned_historical_data/"


# Function to load and preprocess data
@st.cache_data
def load_data(base_path, filters, _fs): # Reverted to using filters
    """
    Loads the partitioned Parquet data from a GCS base path with optional filters,
    using service account credentials if available in Streamlit secrets.
    """
    try:
        # pandas.read_parquet can read partitioned datasets and apply filters
        # Pass the initialized filesystem object
        df = pd.read_parquet(base_path[5:], filters=filters, filesystem=_fs,engine="pyarrow")
        
        # Assuming column names are 'start_time', 'end_time', 'start_station_id', 'end_station_id'
        # Adjust these column names if yours are different
        df['Fecha_Arribo'] = pd.to_datetime(df['Fecha_Arribo'], errors='coerce')
        df['Fecha_Retiro'] = pd.to_datetime(df['Fecha_Retiro'], errors='coerce')
        df['date_start'] = pd.to_datetime(df['date_start'], errors='coerce')
        df['date_end'] = pd.to_datetime(df['date_end'], errors='coerce')

        # Drop rows where time parsing failed
        df.dropna(subset=['date_end', 'date_start','Fecha_Arribo','Fecha_Retiro'], inplace=True)

        df['trip_duration_minutes'] = df['duration'] / 60
        duration_bins = list(np.arange(0,121,5))+[np.inf]
        duration_labels = [f"[{i}-{i+5})" for i in duration_bins[:-2] ] + ["120+"]
        df['trip_duration_min_bucket'] = pd.cut(df['trip_duration_minutes'],bins=duration_bins,labels=duration_labels,right=False)
        df['trip_duration_min_bucket'] = pd.Categorical(df['trip_duration_min_bucket'], categories=duration_labels, ordered=True)

        # Extract time-based features
        df['start_day_of_week'] = df['date_start'].dt.day_name()
        df['end_day_of_week'] = df['date_end'].dt.day_name()


        # Order days of the week for consistent plotting
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['start_day_of_week'] = pd.Categorical(df['start_day_of_week'], categories=day_order, ordered=True)
        df['end_day_of_week'] = pd.Categorical(df['end_day_of_week'], categories=day_order, ordered=True)
        

        return df
    except Exception as e:
        st.error(f"Error loading data from GCS: {e}. Please ensure the GCS path `{base_path}` is the **root** of your partitioned dataset (i.e., contains `year=YYYY/` folders), and that the partitioning scheme (`year=YYYY/month=M/` or `year=YYYY/month=MM/`) matches. Also, verify your service account has `Storage Object Viewer` role.")
        return pd.DataFrame() # Return empty DataFrame on error
    
@st.cache_data
def load_stations_data():
    ECOBICI_API = "https://gbfs.mex.lyftbikes.com/gbfs/gbfs.json"
    response = requests.get(ECOBICI_API,timeout=2)
    api_urls = {x['name']: x['url'] for x in response.json()['data']['en']['feeds']}
    response = requests.get(api_urls['station_information'],timeout=3.1)
    stations = pd.DataFrame.from_dict(response.json()['data']['stations'])
    stations['station_id'] = stations['station_id'].str.lstrip("0")
    # stations.set_index("short_name",inplace=True)
    stations.set_index("station_id",inplace=True)
    
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
    # sns.barplot(x='difference', y='hour', data=d_plot_hour, 
    #             palette=[palette[0] if x>0 else palette[-1] for x in d_plot_hour.difference], ax=ax_imb_hour)
    ax_imb_hour.set_title('Ride Imbalance by Hour')
    ax_imb_hour.axvline(0,color=text_color,linestyle="--",lw=2)
    ax_imb_hour.set_xlabel('Trips Difference (Arrivals - Departures)')
    ax_imb_hour.set_ylabel('Hour')
    st.pyplot(fig_imb_hour)


# Main Streamlit app
def main():
    st.set_page_config(layout="wide", page_title="EcoBici EDAV", page_icon=":bike:")

    st.title(":bike: Mexico City Bike Sharing Exploratory Data Analysis")
    st.write("Select a year and optionally a month to load bike sharing trip data from your Google Cloud Storage bucket.")

    # --- Path Validation Check ---
    if not BASE_GCS_DATA_URL.startswith("gs://"):
        st.error("Configuration Error: `BASE_GCS_DATA_URL` must start with `gs://` for Google Cloud Storage paths.")
        st.stop() # Stop the app execution if the base URL is invalid
    # --- End Path Validation Check ---

    st.sidebar.header("Data Selection")

    # Year selection (required)
    selected_year = st.sidebar.selectbox(
        "Select Year (Required)",
        options=list(range(2019, 2026)), # Years from 2019 to 2025
        index=None, # No default selection
        placeholder="Choose a year..."
    )

    # Month selection (optional)
    month_options = {
        "Select a Month (Optional)": None,
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12
    }
    selected_month_name = st.sidebar.selectbox(
        "Select Month (Optional)",
        options=list(month_options.keys()),
        index=0 # Default to "Select a Month (Optional)"
    )
    selected_month_num = month_options[selected_month_name]

    data = pd.DataFrame() # Initialize an empty DataFrame
    
    # Initialize GCSFileSystem with credentials from Streamlit secrets
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
        # Construct filters for partitioned dataset
        filters = [('year', '=', selected_year)]
        if selected_month_num is not None:
            filters.append(('month', '=', selected_month_num))
        
        # Call load_data with the base path and constructed filters
        data = load_data(BASE_GCS_DATA_URL, filters, fs) # Pass filters and fs

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

        # Sidebar for filters (keeping existing date range filter for loaded data)
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
            filtered_data = data # No date range selected, use all data

        st.sidebar.write(f"Trips in selected range: {len(filtered_data):,.0f}")

        if filtered_data.empty:
            st.warning("No data available for the selected date range. Please adjust your filters.")
            return

        st.write("---") # Separator

        # --- Visualizations ---
        st.header("📊 Key Visualizations")

        # Row 1: Trips Over Time & Day of Week
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Trips Over Time")
            # Aggregate by date
            trips_by_date = filtered_data.groupby(pd.Grouper(key="date_end",freq="D")).size().reset_index(name='trip_count')
            fig_time, ax_time = plt.subplots(figsize=(10, 5))
            ax_time.plot(trips_by_date['date_end'], trips_by_date['trip_count'], color=palette[-1],lw=3)
            ax_time.set_title(f'Daily Trip Counts for Selected Data')
            ax_time.set_xlabel('Date')
            ax_time.set_ylabel('Number of Trips')
            # Format x-axis for dates
            ax_time.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            fig_time.autofmt_xdate()
            st.pyplot(fig_time)

        with col2:
            st.subheader("Trips by Day of Week")
            trips_by_day = filtered_data['start_day_of_week'].value_counts().sort_index().reset_index(name='trip_count')
            trips_by_day.columns = ['day_of_week', 'trip_count'] # Rename columns for clarity
            fig_day, ax_day = plt.subplots(figsize=(10, 5))
            sns.barplot(x='day_of_week', y='trip_count', data=trips_by_day, color=palette[0], ax=ax_day)
            ax_day.set_title('Trip Counts by Day of Week')
            ax_day.set_xlabel('Day of Week')
            ax_day.set_ylabel('Number of Trips')
            st.pyplot(fig_day)

        # Row 2: Trips by Hour & Trip Duration
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
            st.pyplot(fig_hour)

        with col4:
            st.subheader("Trip Duration Distribution")
            fig_duration, ax_duration = plt.subplots(figsize=(10, 5))
            # Filter out very long trips that might skew the histogram
            trips_by_dur = filtered_data.groupby('trip_duration_min_bucket',observed=False).size().reset_index(name='trip_count')
            trips_by_dur.columns=['duration','trip_count']
            sns.barplot(x='duration',y='trip_count',data=trips_by_dur, color=palette[2], ax=ax_duration)
            ax_duration.set_title('Distribution of Trip Durations (minutes)')
            ax_duration.set_xlabel('Trip Duration (minutes)')
            ax_duration.set_ylabel('Frequency')
            ax_duration.set_xticks(np.arange(-0.5,24.5,1))
            ax_duration.set_xticklabels(list(np.arange(0,120,5))+["120+"])
            st.pyplot(fig_duration)

        # Row 3: Top Stations
        col5, col6 = st.columns(2)
        stations = load_stations_data()
        st.write(filtered_data['Ciclo_Estacion_Retiro'].value_counts().to_frame().assign(station_name=lambda x: x.index.map(stations.name)).loc[lambda x: x.station_name.isna()])
        with col5:
            st.subheader("Top 10 Start Stations")
            top_start_stations = filtered_data['Ciclo_Estacion_Retiro'].value_counts().head(10).to_frame(name='trip_count')
            top_start_stations['station_name'] = top_start_stations.index.map(stations.name)
            st.write(top_start_stations)
            fig_start_station, ax_start_station = plt.subplots(figsize=(10, 10))
            sns.barplot(x='trip_count', y='station_name', data=top_start_stations, color=palette[0], ax=ax_start_station)
            ax_start_station.set_title('Top 10 Start Stations by Trip Count')
            ax_start_station.set_xlabel('Number of Trips')
            ax_start_station.set_ylabel('Station ID')
            st.pyplot(fig_start_station)

        with col6:
            st.subheader("Top 10 End Stations")
            top_end_stations = filtered_data['Ciclo_Estacion_Arribo'].value_counts().head(10).to_frame(name='trip_count')
            top_end_stations['station_name'] = top_end_stations.index.map(stations.name)
            fig_end_station, ax_end_station = plt.subplots(figsize=(10, 10))
            sns.barplot(x='trip_count', y='station_name', data=top_end_stations, color=palette[2], ax=ax_end_station)
            ax_end_station.set_title('Top 10 End Stations by Trip Count')
            ax_end_station.set_xlabel('Number of Trips')
            ax_end_station.set_ylabel('Station ID')
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
                        palette=[palette[0] if x>0 else palette[-1] for x in d_plot.difference], ax=ax_imb)
            ax_imb.set_title('Ride Imbalance by Station')
            ax_imb.axvline(0,color=text_color,linestyle="--",lw=2)
            ax_imb.set_xlabel('Trips Difference (Arrivals - Departures)')
            ax_imb.set_ylabel('Station')
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
        st.write("### Data Source: [Mexico City Ecobici](https://www.ecobici.cdmx.gob.mx/)")

    else:
        st.warning("No data loaded.")

if __name__ == "__main__":
    main()
