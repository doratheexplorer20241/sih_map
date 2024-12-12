import streamlit as st
from streamlit_lottie import st_lottie
import pandas as pd
import requests
import numpy as np
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import folium
from streamlit_folium import st_folium
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import requests
import json

# Initialize session state variables
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'map_data' not in st.session_state:
    st.session_state.map_data = None
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'sort_by' not in st.session_state:
    st.session_state.sort_by = 'KNN_Score'
if 'sort_ascending' not in st.session_state:
    st.session_state.sort_ascending = False
if 'selected_lawyer' not in st.session_state:
    st.session_state.selected_lawyer = None
if 'user_location' not in st.session_state:
    st.session_state.user_location = None

def geocode_address(address):
    """
    Convert address to coordinates using Nominatim geocoding service
    Args:
        address (str): Address string to geocode
    Returns:
        tuple: (latitude, longitude) or None if geocoding fails
    """
    try:
        geolocator = Nominatim(user_agent="lawyer_recommender")
        location = geolocator.geocode(address)
        if location:
            return (location.latitude, location.longitude)
        return None
    except GeocoderTimedOut:
        st.error("Geocoding service timed out. Please try again.")
        return None
    except Exception as e:
        st.error(f"Error geocoding address: {str(e)}")
        return None

@st.cache_data
def load_data():
    """Load and cache the lawyers and offense categories data"""
    try:
        lawyers_data = pd.read_csv('lawyers_dataset_2km.csv')
        offense_categories = pd.read_csv('Law_Offenses_Dataset.csv')
        return lawyers_data, offense_categories
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def get_field_of_law(offense, offense_categories):
    """Get the field of law for a given offense"""
    field = offense_categories[offense_categories['Offense'] == offense]['Field of Law'].iloc[0]
    return field

def get_route_info(origin, destination):
    """
    Get route information using OSRM
    Args:
        origin (tuple): (latitude, longitude) of starting point
        destination (tuple): (latitude, longitude) of destination
    Returns:
        dict: Route information including path, duration, and distance
    """
    try:
        # Format coordinates for OSRM
        coords = f"{origin[1]},{origin[0]};{destination[1]},{destination[0]}"
        
        # Make request to OSRM routing service
        url = f"https://router.project-osrm.org/route/v1/driving/{coords}"
        params = {
            "overview": "full",
            "geometries": "geojson",
            "steps": "true"
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        route_data = response.json()
        
        if "routes" in route_data and len(route_data["routes"]) > 0:
            route = route_data["routes"][0]
            coordinates = route["geometry"]["coordinates"]
            
            # Convert coordinates from [lon, lat] to [lat, lon] for folium
            path = [[coord[1], coord[0]] for coord in coordinates]
            
            # Extract duration and distance
            duration_minutes = round(route["duration"] / 60)
            distance_km = round(route["distance"] / 1000, 2)
            
            # Extract turn-by-turn instructions if available
            instructions = []
            if "legs" in route and len(route["legs"]) > 0:
                for step in route["legs"][0]["steps"]:
                    instructions.append(step.get("maneuver", {}).get("instruction", ""))
            
            return {
                "path": path,
                "duration": f"{duration_minutes} minutes",
                "distance": f"{distance_km} km",
                "instructions": instructions,
                "success": True
            }
        else:
            return {"success": False}
            
    except Exception as e:
        st.error(f"Error fetching route: {str(e)}")
        return {"success": False}

@st.cache_data
def prepare_features(lawyers_df_json, user_location=None):
    """Prepare and scale features for KNN algorithm"""
    lawyers_df = pd.read_json(lawyers_df_json)
    lawyers_df['Success Rate'] = (lawyers_df['Number of Wins'] / lawyers_df['Total Cases']) * 100
    features = ['Success Rate', 'Per Case Fee (INR)', 'Total Cases']
    X = lawyers_df[features].copy()
    
    if user_location:
        lawyers_df['Distance'] = lawyers_df.apply(
            lambda x: geodesic(user_location, (x['Latitude'], x['Longitude'])).kilometers,
            axis=1
        )
        X['Distance'] = lawyers_df['Distance']
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns), lawyers_df

@st.cache_data
def get_lawyer_recommendations_knn(_lawyers_data_json, offense, offense_categories, user_location=None, k=5):
    """Get lawyer recommendations using KNN algorithm"""
    lawyers_data = pd.read_json(_lawyers_data_json)
    field_of_law = get_field_of_law(offense, offense_categories)
    relevant_lawyers = lawyers_data[lawyers_data['Field of Law'] == field_of_law].copy()
    
    if relevant_lawyers.empty:
        return pd.DataFrame()
    
    X_scaled, relevant_lawyers = prepare_features(relevant_lawyers.to_json(), user_location)
    ideal_profile = np.ones(X_scaled.shape[1])
    ideal_profile[1] = 0  # Lower fee is better
    if user_location:
        ideal_profile[3] = 0  # Shorter distance is better
    
    knn = NearestNeighbors(n_neighbors=min(k, len(relevant_lawyers)), metric='euclidean')
    knn.fit(X_scaled)
    distances, indices = knn.kneighbors([ideal_profile])
    recommended_lawyers = relevant_lawyers.iloc[indices[0]]
    recommended_lawyers['KNN_Score'] = 1 / (1 + distances[0])
    return recommended_lawyers

def create_map(lawyers_df, user_location=None):
    """Create an interactive map with lawyers and routes"""
    center_lat = user_location[0] if user_location else 20.5937
    center_lon = user_location[1] if user_location else 78.9629
    
    m = folium.Map(location=[center_lat, center_lon], 
                   zoom_start=6 if user_location else 5,
                   tiles="OpenStreetMap")
    
    lawyers_layer = folium.FeatureGroup(name="Lawyers")
    routes_layer = folium.FeatureGroup(name="Routes")
    
    for idx, lawyer in lawyers_df.iterrows():
        success_rate = (lawyer['Number of Wins'] / lawyer['Total Cases'] * 100)
        score = lawyer.get('KNN_Score', 0)
        color = 'red' if score > 0.8 else 'orange' if score > 0.6 else 'blue'
        
        folium.Marker(
            [lawyer['Latitude'], lawyer['Longitude']],
            popup=folium.Popup(f"""
            <div style='min-width: 200px'>
                <h4>{lawyer['Lawyer Name']}</h4>
                <b>Field:</b> {lawyer['Field of Law']}<br>
                <b>Success Rate:</b> {success_rate:.1f}%<br>
                <b>Cases:</b> {lawyer['Number of Wins']}/{lawyer['Total Cases']}<br>
                <b>Fee:</b> ₹{lawyer['Per Case Fee (INR)']:,.2f}<br>
                <b>Score:</b> {score:.3f}<br>
                <b>Contact:</b> {lawyer['Mobile']}<br>
                <b>Email:</b> {lawyer['Email']}
            </div>
            """, max_width=300),
            tooltip=lawyer['Lawyer Name'],
            icon=folium.Icon(color=color, icon='info-sign')
        ).add_to(lawyers_layer)
        
        if user_location and st.session_state.selected_lawyer is not None and idx == st.session_state.selected_lawyer:
            lawyer_location = (lawyer['Latitude'], lawyer['Longitude'])
            route_info = get_route_info(user_location, lawyer_location)
            
            if route_info["success"]:
                folium.PolyLine(
                    locations=route_info["path"],
                    weight=4,
                    color='red',
                    opacity=0.8,
                    tooltip=f"Route to {lawyer['Lawyer Name']}"
                ).add_to(routes_layer)
                
                mid_point = [(user_location[0] + lawyer['Latitude'])/2, 
                            (user_location[1] + lawyer['Longitude'])/2]
                
                route_info_html = f"""
                <div>
                    <b>Distance:</b> {route_info['distance']}<br>
                    <b>Est. Duration:</b> {route_info['duration']}<br>
                    <b>Directions:</b><br>
                    <ul>
                """
                
                for instruction in route_info['instructions']:
                    route_info_html += f"<li>{instruction}</li>"
                
                route_info_html += "</ul></div>"
                
                folium.Popup(route_info_html, max_width=300).add_to(routes_layer)
    
    if user_location:
        folium.Marker(
            user_location,
            popup="Your Location",
            icon=folium.Icon(color='green', icon='home'),
            tooltip="Your Location"
        ).add_to(lawyers_layer)
    
    lawyers_layer.add_to(m)
    routes_layer.add_to(m)
    folium.LayerControl().add_to(m)
    return m

def get_recommendations():
    """Get lawyer recommendations based on user input"""
    st.session_state.recommendations = get_lawyer_recommendations_knn(
        st.session_state.lawyers_data.to_json(),
        st.session_state.offense,
        st.session_state.offense_categories,
        st.session_state.user_location,
        st.session_state.k
    )
    st.session_state.show_results = True

def load_lottie(url: str):
    """Load Lottie animation from URL"""
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

def main():
    """Main application function"""
    # Custom CSS
    st.markdown("""
        <style>
        .classic-heading {
            color: #1a1a1a;
            text-align: center;
            padding: 2rem 0;
            margin-bottom: 2rem;
            border-bottom: 2px solid #1a1a1a;
            font-family: "Times New Roman", Times, serif;
        }
        .classic-heading h1 {
            font-size: 2.5rem;
            font-weight: normal;
            margin-bottom: 0.5rem;
            letter-spacing: 0.05em;
        }
        .classic-heading p {
            font-size: 1.2rem;
            color: #4a4a4a;
            font-style: italic;
        }
        .classic-heading img {
            width: 40px;
            height: 40px;
            margin-right: 10px;
            vertical-align: middle;
        }
        /* Rest of your CSS styles */
        .main {
            padding: 0rem 2rem;
        }
        .stButton>button {
            width: 100%;
            background-color: #0066cc;
            color: white;
            border-radius: 5px;
            height: 3em;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #0052a3;
            border-color: #0052a3;
        }
        .big-font {
            font-size: 24px !important;
            font-weight: bold;
        }
        .info-box {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .lawyer-card {
            border: 1px solid #e6e6e6;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .success-box {
            padding: 1rem;
            border-radius: 5px;
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .error-box {
            padding: 1rem;
            border-radius: 5px;
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        .heading-container {
            background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Lawyer Recommendation System")
    
    lawyers_data, offense_categories = load_data()
    st.session_state.lawyers_data = lawyers_data
    st.session_state.offense_categories = offense_categories
    
    if lawyers_data.empty or offense_categories.empty:
        st.error("Unable to load data. Please check if the CSV files are present and properly formatted.")
        return
    
    st.markdown("""
        <div class="info-box">
            <p class="big-font">🎯 Find Your Legal Representative</p>
            <p>Select your case details and location to get personalized recommendations</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📋 Case Details")
    st.session_state.offense = st.selectbox(
        "Select the type of offense:",
        options=offense_categories['Offense'].unique(),
        key='offense_select'
    )
    
    selected_field = get_field_of_law(st.session_state.offense, offense_categories)
    st.info(f"Field of Law: {selected_field}")
    
    st.markdown("### 📍 Location")
    use_location = st.checkbox("Include my location for better recommendations", 
                             help="Enable this to find lawyers near you")
    
    if use_location:
        st.markdown("""
            <div class="info-box" style="background-color: #e8f4f8;">
                <p>Choose how you want to share your location:</p>
            </div>
        """, unsafe_allow_html=True)
        location_type = st.radio(
            "How would you like to input your location?",
            ["Enter Address", "Enter Coordinates"],
            key="location_type"
        )
        
        if location_type == "Enter Address":
            address = st.text_input("Enter your address", key="address_input",
                                  placeholder="Enter full address (e.g., 123 Main St, City, State, Country)")
            if address:
                coordinates = geocode_address(address)
                if coordinates:
                    st.success(f"Address found! Coordinates: {coordinates[0]:.4f}, {coordinates[1]:.4f}")
                    st.session_state.user_location = coordinates
                else:
                    st.error("Could not find coordinates for this address. Try being more specific or use coordinate input instead.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                latitude = st.number_input("Your Latitude", value=28.6139)
            with col2:
                longitude = st.number_input("Your Longitude", value=77.2090)
            st.session_state.user_location = (latitude, longitude)
    else:
        st.session_state.user_location = None
    
    st.session_state.k = st.slider(
        "Number of recommendations", 
        min_value=1, 
        max_value=10, 
        value=5,
        key='k_slider'
    )
    
    if st.button("Get Recommendations", key='rec_button'):
        get_recommendations()
    
    if st.session_state.show_results and st.session_state.recommendations is not None:
        if st.session_state.recommendations.empty:
            st.warning(f"No lawyers found for {selected_field}. Please try a different offense type.")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Sort by KNN Score", key="sort_knn"):
                    st.session_state.sort_by = 'KNN_Score'
                    st.session_state.sort_ascending = False
            with col2:
                if st.button("Sort by Price", key="sort_price"):
                    st.session_state.sort_by = 'Per Case Fee (INR)'
                    st.session_state.sort_ascending = True
            with col3:
                if st.session_state.user_location:
                    if st.button("Sort by Distance", key="sort_distance"):
                        st.session_state.sort_by = 'Distance'
                        st.session_state.sort_ascending = True
            
            if st.session_state.sort_by == 'Distance' and st.session_state.user_location:
                st.session_state.recommendations['Distance'] = st.session_state.recommendations.apply(
                    lambda x: geodesic(st.session_state.user_location, (x['Latitude'], x['Longitude'])).kilometers,
                    axis=1
                )
            
            sorted_recommendations = st.session_state.recommendations.sort_values(
                by=st.session_state.sort_by,
                ascending=st.session_state.sort_ascending
            )
            
            st.subheader("Lawyer Locations")
            map_obj = create_map(sorted_recommendations, st.session_state.user_location)
            st_folium(map_obj, width=800, height=500, returned_objects=[])
            
            if st.session_state.user_location:
                st.subheader("Show Route to Lawyer")
                selected_lawyer_name = st.selectbox(
                    "Select a lawyer to show route:",
                    options=sorted_recommendations['Lawyer Name'].tolist(),
                    key='route_selector'
                )
                if st.button("Show Route"):
                    lawyer_idx = sorted_recommendations[sorted_recommendations['Lawyer Name'] == selected_lawyer_name].index[0]
                    st.session_state.selected_lawyer = lawyer_idx
                    st.rerun()
            
            st.markdown("""
                <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 10px; margin: 2rem 0;'>
                    <h2 style='color: #1e3c72;'>🎯 Top Recommended Lawyers</h2>
                    <p>Here are the best-matched lawyers for your case, ranked by our recommendation algorithm</p>
                </div>
            """, unsafe_allow_html=True)
            
            for idx, lawyer in sorted_recommendations.iterrows():
                st.markdown(f"""
                    <div class="lawyer-card">
                        <h3 style="color: #1e3c72; margin-bottom: 1rem;">
                            {lawyer['Lawyer Name']} 
                            <span style="float: right; background-color: #e8f4f8; padding: 5px 10px; border-radius: 15px; font-size: 0.8em;">
                                Match Score: {lawyer['KNN_Score']:.3f}
                            </span>
                        </h3>
                    </div>
                """, unsafe_allow_html=True)
                
                with st.expander("View Details", expanded=(idx == sorted_recommendations.index[0])):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("*Contact Information:*")
                        st.write(f"📱 Mobile: {lawyer['Mobile']}")
                        st.write(f"📧 Email: {lawyer['Email']}")
                    with col2:
                        st.write("*Experience:*")
                        st.write(f"Total Cases: {lawyer['Total Cases']}")
                        st.write(f"Wins: {lawyer['Number of Wins']}")
                        st.write(f"Success Rate: {(lawyer['Number of Wins']/lawyer['Total Cases']*100):.1f}%")
                        st.write(f"Fee per Case: ₹{lawyer['Per Case Fee (INR)']:,.2f}")
                    st.write("*Profile:*")
                    st.write(lawyer['Profile'])
                    if st.session_state.user_location:
                        distance = geodesic(
                            st.session_state.user_location,
                            (lawyer['Latitude'], lawyer['Longitude'])
                        ).kilometers
                        st.write(f"📍 Distance: {distance:.1f} km")

if _name_ == "_main_":
    main()
