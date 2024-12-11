import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import folium
from streamlit_folium import st_folium
from geopy.distance import geodesic

# Initialize session state
if 'lawyers_data' not in st.session_state:
    st.session_state.lawyers_data = None
if 'offense_categories' not in st.session_state:
    st.session_state.offense_categories = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'map_data' not in st.session_state:
    st.session_state.map_data = None

@st.cache_data(ttl=3600)
def load_data():
    """Load and preprocess the data"""
    try:
        lawyers_data = pd.read_csv('Indian_Lawyers_Dataset.csv')
        offense_categories = pd.read_csv('Law_Offenses_Dataset.csv')
        
        # Calculate success rate
        lawyers_data['Success Rate'] = (lawyers_data['Number of Wins'] / lawyers_data['Total Cases']) * 100
        
        return lawyers_data, offense_categories
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

def get_relevant_lawyers_knn(offense, lawyers_data, offense_categories):
    """Use KNN to find relevant lawyers for the given offense"""
    try:
        # Extract field of law for the offense
        offense_info = offense_categories[offense_categories['Offense'] == offense].iloc[0]
        field_of_law = offense_info['Field of Law']
        
        # Filter lawyers by field of law
        relevant_lawyers = lawyers_data[lawyers_data['Field of Law'] == field_of_law]
        
        if relevant_lawyers.empty:
            return None

        # Features for KNN
        X = relevant_lawyers[['Success Rate', 'Total Cases']]
        y = relevant_lawyers.index

        # Use KNN to rank lawyers
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X, y)
        distances, indices = knn.kneighbors(X)

        # Extract top lawyers based on KNN
        top_lawyers = relevant_lawyers.iloc[indices[0]]
        return top_lawyers
    except Exception as e:
        st.error(f"Error with KNN calculation: {str(e)}")
        return None

def create_map(recommended_lawyers, user_location=None):
    """Create a map with lawyer and user locations"""
    try:
        # Map center
        if user_location:
            center_lat, center_lon = user_location
        else:
            center_lat = recommended_lawyers['Latitude'].mean()
            center_lon = recommended_lawyers['Longitude'].mean()

        # Create map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

        # Add lawyer markers
        for _, lawyer in recommended_lawyers.iterrows():
            folium.Marker(
                [lawyer['Latitude'], lawyer['Longitude']],
                popup=f"{lawyer['Lawyer Name']}",
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(m)

        # Add user location marker
        if user_location:
            folium.Marker(
                [user_location[0], user_location[1]],
                popup="Your Location",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)

        return m
    except Exception as e:
        st.error(f"Error creating map: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Lawyer Recommendation System", layout="wide")
    st.title("Lawyer Recommendation System")

    # Load data
    if st.session_state.lawyers_data is None or st.session_state.offense_categories is None:
        st.session_state.lawyers_data, st.session_state.offense_categories = load_data()

    if st.session_state.lawyers_data is None or st.session_state.offense_categories is None:
        st.error("Failed to load data. Please check if the CSV files are present.")
        return

    # User inputs
    offense = st.selectbox(
        "Select the type of offense:",
        options=st.session_state.offense_categories['Offense'].unique(),
        key='offense_select'
    )

    # Display offense information
    offense_info = st.session_state.offense_categories[
        st.session_state.offense_categories['Offense'] == offense
    ].iloc[0]
    # st.info(f"Field of Law: {offense_info['Field of Law']} | Bailable: {offense_info['Bailable']}")

    use_location = st.checkbox("Include my location for better recommendations", key='use_location')
    user_location = None

    if use_location:
        col1, col2 = st.columns(2)
        with col1:
            latitude = st.number_input("Your Latitude", value=28.6139, key='lat_input')
        with col2:
            longitude = st.number_input("Your Longitude", value=77.2090, key='lon_input')
        user_location = (latitude, longitude)

    # Get recommendations button
    if st.button("Get Recommendations", key='get_recommendations'):
        with st.spinner("Finding the best lawyers for your case..."):
            recommendations = get_relevant_lawyers_knn(
                offense,
                st.session_state.lawyers_data,
                st.session_state.offense_categories
            )

            if recommendations is None or recommendations.empty:
                st.warning("No lawyers found matching your criteria.")
            else:
                st.session_state.recommendations = recommendations
                st.session_state.map_data = create_map(recommendations, user_location)

    # Display results if available
    if st.session_state.recommendations is not None and st.session_state.map_data is not None:
        st.subheader("Lawyer Locations")
        st_folium(st.session_state.map_data, width=800, height=600)

        col_left, col_right = st.columns([3, 1])

        with col_left:
            st.subheader("Recommended Lawyers")
        with col_right:
            if st.button("Sort by Price (Ascending)",type='primary'):
                st.session_state.recommendations = st.session_state.recommendations.sort_values(by='Per Case Fee (INR)')
            if st.button("Sort by Price (Descending)",type='primary'):
                st.session_state.recommendations = st.session_state.recommendations.sort_values(by='Per Case Fee (INR)', ascending=False)

        for _, lawyer in st.session_state.recommendations.iterrows():
            with st.expander(f"{lawyer['Lawyer Name']} - Success Rate: {lawyer['Success Rate']:.1f}%"):
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Contact Information:**")
                    st.write(f"📱 Mobile: {lawyer['Mobile']}")
                    st.write(f"📧 Email: {lawyer['Email']}")

                with col2:
                    st.write("**Experience:**")
                    st.write(f"Total Cases: {lawyer['Total Cases']}")
                    st.write(f"Wins: {lawyer['Number of Wins']}")
                    st.write(f"Fee per Case: ₹{lawyer['Per Case Fee (INR)']:,.2f}")

                st.write("**Profile:**")
                st.write(lawyer['Profile'])

                st.write("**Cases Handled:**")
                st.write(lawyer['Cases Handled'])

                if user_location and 'Distance' in lawyer:
                    st.write(f"📍 Distance: {lawyer['Distance']:.1f} km")

if __name__ == "__main__":
    main()
