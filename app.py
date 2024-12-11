import streamlit as st
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium

# Load the data
@st.cache_data
def load_data():
    try:
        # Load lawyers data from CSV
        lawyers_data = pd.read_csv('Indian_Lawyers_Dataset.csv')
        
        # Load offense categories from CSV
        offense_categories = pd.read_csv('Law_Offenses_Dataset.csv')
        
        return lawyers_data, offense_categories
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def calculate_success_rate(row):
    return (row['Number of Wins'] / row['Total Cases']) * 100

def create_map(lawyers_df, user_location=None):
    # Create a map centered on India
    center_lat = 20.5937
    center_lon = 78.9629
    if user_location:
        center_lat, center_lon = user_location
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=5)
    
    # Add markers for each lawyer
    for idx, lawyer in lawyers_df.iterrows():
        popup_text = f"""
        <b>{lawyer['Lawyer Name']}</b><br>
        Field: {lawyer['Field of Law']}<br>
        Success Rate: {(lawyer['Number of Wins']/lawyer['Total Cases']*100):.1f}%<br>
        Cases Won: {lawyer['Number of Wins']}/{lawyer['Total Cases']}<br>
        Fee: ₹{lawyer['Per Case Fee (INR)']:,.2f}
        """
        
        folium.Marker(
            [lawyer['Latitude'], lawyer['Longitude']],
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=lawyer['Lawyer Name']
        ).add_to(m)
    
    # Add user location if provided
    if user_location:
        folium.Marker(
            user_location,
            popup="Your Location",
            icon=folium.Icon(color='red', icon='info-sign'),
            tooltip="Your Location"
        ).add_to(m)
    
    return m

def get_lawyer_recommendations(offense, user_location=None):
    lawyers_data, _ = load_data()
    
    # Filter lawyers based on offense category field of law
    field_of_law = 'Criminal Law'  # We can expand this based on offense categories mapping
    relevant_lawyers = lawyers_data[lawyers_data['Field of Law'] == field_of_law].copy()
    
    if relevant_lawyers.empty:
        return pd.DataFrame()
    
    # Calculate success rate
    relevant_lawyers['Success Rate'] = relevant_lawyers.apply(calculate_success_rate, axis=1)
    
    # Calculate distance if user location is provided
    if user_location:
        relevant_lawyers['Distance'] = relevant_lawyers.apply(
            lambda x: geodesic(user_location, (x['Latitude'], x['Longitude'])).kilometers,
            axis=1
        )
        
        # Normalize factors
        max_distance = relevant_lawyers['Distance'].max()
        min_distance = relevant_lawyers['Distance'].min()
        max_fee = relevant_lawyers['Per Case Fee (INR)'].max()
        min_fee = relevant_lawyers['Per Case Fee (INR)'].min()
        
        # Calculate composite score (higher is better)
        relevant_lawyers['Score'] = (
            relevant_lawyers['Success Rate'] * 0.4 +
            ((max_fee - relevant_lawyers['Per Case Fee (INR)']) / (max_fee - min_fee)) * 0.3 +
            ((max_distance - relevant_lawyers['Distance']) / (max_distance - min_distance)) * 0.3
        )
    else:
        # Without location, only consider success rate and fees
        max_fee = relevant_lawyers['Per Case Fee (INR)'].max()
        min_fee = relevant_lawyers['Per Case Fee (INR)'].min()
        
        relevant_lawyers['Score'] = (
            relevant_lawyers['Success Rate'] * 0.6 +
            ((max_fee - relevant_lawyers['Per Case Fee (INR)']) / (max_fee - min_fee)) * 0.4
        )
    
    # Sort by score and return top 5
    return relevant_lawyers.nlargest(5, 'Score')

def main():
    st.title("Lawyer Recommendation System")
    
    lawyers_data, offense_categories = load_data()
    
    if lawyers_data.empty or offense_categories.empty:
        st.error("Unable to load data. Please check if the CSV files are present and properly formatted.")
        return
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Find a Lawyer", "View All Lawyers"])
    
    with tab1:
        st.header("Find a Lawyer")
        
        # Offense selection
        offense = st.selectbox(
            "Select the type of offense:",
            options=offense_categories['Offense'].unique()
        )
        
        # Optional location input
        use_location = st.checkbox("Include my location for better recommendations")
        user_location = None
        
        if use_location:
            col1, col2 = st.columns(2)
            with col1:
                latitude = st.number_input("Your Latitude", value=28.6139)
            with col2:
                longitude = st.number_input("Your Longitude", value=77.2090)
            user_location = (latitude, longitude)
        
        if st.button("Get Recommendations"):
            recommendations = get_lawyer_recommendations(offense, user_location)
            
            if recommendations.empty:
                st.warning("No lawyers found matching your criteria.")
            else:
                # Display map with recommended lawyers
                st.subheader("Lawyer Locations")
                map_data = create_map(recommendations, user_location)
                st_folium(map_data, width=800, height=400)
                
                st.subheader("Recommended Lawyers")
                
                for idx, lawyer in recommendations.iterrows():
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
                        
                        if user_location:
                            distance = geodesic(
                                user_location,
                                (lawyer['Latitude'], lawyer['Longitude'])
                            ).kilometers
                            st.write(f"📍 Distance: {distance:.1f} km")
    
    with tab2:
        st.header("All Lawyers")
        
        # Display map with all lawyers
        st.subheader("Lawyer Locations")
        map_data = create_map(lawyers_data)
        st_folium(map_data, width=800, height=400)
        
        # Display all lawyers in a table
        st.subheader("Lawyer Details")
        display_cols = ['Lawyer Name', 'Field of Law', 'Number of Wins', 'Total Cases', 
                       'Per Case Fee (INR)', 'Mobile', 'Email']
        st.dataframe(lawyers_data[display_cols])

if __name__ == "__main__":
    main()