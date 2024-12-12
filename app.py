import streamlit as st
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

# Initialize session state
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

# Load the data
@st.cache_data
def load_data():
    try:
        lawyers_data = pd.read_csv('Indian_Lawyers_Dataset.csv')
        offense_categories = pd.read_csv('Law_Offenses_Dataset.csv')
        return lawyers_data, offense_categories
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def get_field_of_law(offense, offense_categories):
    """Map offense to field of law"""
    field = offense_categories[offense_categories['Offense'] == offense]['Field of Law'].iloc[0]
    return field

@st.cache_data
def prepare_features(lawyers_df_json, user_location=None):
    """Prepare and scale features for KNN"""
    lawyers_df = pd.read_json(lawyers_df_json)
    
    # Calculate success rate
    lawyers_df['Success Rate'] = (lawyers_df['Number of Wins'] / lawyers_df['Total Cases']) * 100
    
    # Initialize feature matrix
    features = ['Success Rate', 'Per Case Fee (INR)', 'Total Cases']
    X = lawyers_df[features].copy()
    
    # Add distance feature if user location is provided
    if user_location:
        lawyers_df['Distance'] = lawyers_df.apply(
            lambda x: geodesic(user_location, (x['Latitude'], x['Longitude'])).kilometers,
            axis=1
        )
        X['Distance'] = lawyers_df['Distance']
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return pd.DataFrame(X_scaled, columns=X.columns), lawyers_df

@st.cache_data
def get_lawyer_recommendations_knn(_lawyers_data_json, offense, offense_categories, user_location=None, k=5):
    """Get lawyer recommendations using KNN"""
    lawyers_data = pd.read_json(_lawyers_data_json)
    
    # Get appropriate field of law for the offense
    field_of_law = get_field_of_law(offense, offense_categories)
    
    # Filter lawyers based on field of law
    relevant_lawyers = lawyers_data[lawyers_data['Field of Law'] == field_of_law].copy()
    
    if relevant_lawyers.empty:
        return pd.DataFrame()
    
    # Prepare features for KNN
    X_scaled, relevant_lawyers = prepare_features(relevant_lawyers.to_json(), user_location)
    
    # Create ideal lawyer profile
    ideal_profile = np.ones(X_scaled.shape[1])
    ideal_profile[1] = 0  # Prefer lower fees
    if user_location:
        ideal_profile[3] = 0  # Prefer closer distance
    
    # Fit KNN model
    knn = NearestNeighbors(n_neighbors=min(k, len(relevant_lawyers)), metric='euclidean')
    knn.fit(X_scaled)
    
    # Find k nearest neighbors to ideal profile
    distances, indices = knn.kneighbors([ideal_profile])
    
    # Get recommended lawyers
    recommended_lawyers = relevant_lawyers.iloc[indices[0]]
    recommended_lawyers['KNN_Score'] = 1 / (1 + distances[0])
    
    return recommended_lawyers

def create_map(lawyers_df, user_location=None):
    """Create an interactive map with lawyer locations"""
    # Create a map centered on India or user location
    center_lat = user_location[0] if user_location else 20.5937
    center_lon = user_location[1] if user_location else 78.9629
    
    m = folium.Map(location=[center_lat, center_lon], 
                   zoom_start=6 if user_location else 5,
                   tiles="OpenStreetMap")
    
    # Add markers for each lawyer
    for idx, lawyer in lawyers_df.iterrows():
        # Calculate success rate
        success_rate = (lawyer['Number of Wins'] / lawyer['Total Cases'] * 100)
        
        # Color marker based on KNN score
        score = lawyer.get('KNN_Score', 0)
        color = 'red' if score > 0.8 else 'orange' if score > 0.6 else 'blue'
        
        popup_text = f"""
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
        """
        
        folium.Marker(
            [lawyer['Latitude'], lawyer['Longitude']],
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=lawyer['Lawyer Name'],
            icon=folium.Icon(color=color, icon='info-sign')
        ).add_to(m)
    
    # Add user location if provided
    if user_location:
        folium.Marker(
            user_location,
            popup="Your Location",
            icon=folium.Icon(color='green', icon='home'),
            tooltip="Your Location"
        ).add_to(m)
    
    # Add layer controls
    folium.LayerControl().add_to(m)
    
    return m

def get_recommendations():
    """Callback function for the Get Recommendations button"""
    st.session_state.recommendations = get_lawyer_recommendations_knn(
        st.session_state.lawyers_data.to_json(),
        st.session_state.offense,
        st.session_state.offense_categories,
        st.session_state.user_location,
        st.session_state.k
    )
    st.session_state.show_results = True

def main():
    st.title("Lawyer Recommendation System")
    
    # Load data
    lawyers_data, offense_categories = load_data()
    st.session_state.lawyers_data = lawyers_data
    st.session_state.offense_categories = offense_categories
    
    if lawyers_data.empty or offense_categories.empty:
        st.error("Unable to load data. Please check if the CSV files are present and properly formatted.")
        return
    
    st.header("Find a Lawyer")
    
    # Offense selection
    st.session_state.offense = st.selectbox(
        "Select the type of offense:",
        options=offense_categories['Offense'].unique(),
        key='offense_select'
    )
    
    # Show field of law based on selected offense
    selected_field = get_field_of_law(st.session_state.offense, offense_categories)
    st.info(f"Field of Law: {selected_field}")
    
    # Optional location input
    use_location = st.checkbox("Include my location for better recommendations")
    st.session_state.user_location = None
    
    if use_location:
        col1, col2 = st.columns(2)
        with col1:
            latitude = st.number_input("Your Latitude", value=28.6139)
        with col2:
            longitude = st.number_input("Your Longitude", value=77.2090)
        st.session_state.user_location = (latitude, longitude)
    
    # Number of recommendations
    st.session_state.k = st.slider(
        "Number of recommendations", 
        min_value=1, 
        max_value=10, 
        value=5,
        key='k_slider'
    )
    
    # Get recommendations button
    if st.button("Get Recommendations", key='rec_button'):
        get_recommendations()
    
    # Display recommendations if available
    if st.session_state.show_results and st.session_state.recommendations is not None:
        if st.session_state.recommendations.empty:
            st.warning(f"No lawyers found for {selected_field}. Please try a different offense type.")
        else:
            # Create and display map
            st.subheader("Lawyer Locations")
            map_obj = create_map(st.session_state.recommendations, st.session_state.user_location)
            st_folium(map_obj, width=800, height=500, returned_objects=[])
            
            st.subheader("Recommended Lawyers")
            
            # Add sorting options
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

            # Initialize sorting if not set
            if 'sort_by' not in st.session_state:
                st.session_state.sort_by = 'KNN_Score'
                st.session_state.sort_ascending = False
                
            # Sort recommendations based on selected criterion
            if st.session_state.sort_by == 'Distance' and st.session_state.user_location:
                # Calculate distances if sorting by distance
                st.session_state.recommendations['Distance'] = st.session_state.recommendations.apply(
                    lambda x: geodesic(st.session_state.user_location, (x['Latitude'], x['Longitude'])).kilometers,
                    axis=1
                )
            
            sorted_recommendations = st.session_state.recommendations.sort_values(
                by=st.session_state.sort_by,
                ascending=st.session_state.sort_ascending
            )
            
            for idx, lawyer in sorted_recommendations.iterrows():
                with st.expander(
                    f"{lawyer['Lawyer Name']} - Score: {lawyer['KNN_Score']:.3f}", 
                    expanded=(idx == 0)  # Expand first recommendation by default
                ):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Contact Information:**")
                        st.write(f"📱 Mobile: {lawyer['Mobile']}")
                        st.write(f"📧 Email: {lawyer['Email']}")
                        
                    with col2:
                        st.write("**Experience:**")
                        st.write(f"Total Cases: {lawyer['Total Cases']}")
                        st.write(f"Wins: {lawyer['Number of Wins']}")
                        st.write(f"Success Rate: {(lawyer['Number of Wins']/lawyer['Total Cases']*100):.1f}%")
                        st.write(f"Fee per Case: ₹{lawyer['Per Case Fee (INR)']:,.2f}")
                    
                    st.write("**Profile:**")
                    st.write(lawyer['Profile'])
                    
                    if st.session_state.user_location:
                        distance = geodesic(
                            st.session_state.user_location,
                            (lawyer['Latitude'], lawyer['Longitude'])
                        ).kilometers
                        st.write(f"📍 Distance: {distance:.1f} km")

if __name__ == "__main__":
    main()
