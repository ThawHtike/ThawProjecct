import streamlit as st
import requests

# Set the title of the app
st.title("Weather Information App")

# User input for city name
city = st.text_input("Enter city name:", "Yangon")

# API Key for OpenWeatherMap
api_key = "a5bedda26f30751cafc720db15d72c20"

# Make a request to the OpenWeatherMap API when the button is clicked
if st.button("Get Weather"):
    # URL to fetch the weather data
    url = f'https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric'

    # Fetch data from the API
    response = requests.get(url)
    data = response.json()

    # Check if the request was successful
    if response.status_code == 200:
        # Display the weather information
        st.subheader(f"Weather in {city.title()}:")
        st.write('Weather: ', data['weather'][0]['description'])
        st.write('Current Temperature: ', data['main']['temp'], '°C')
        st.write('Feels Like: ', data['main']['feels_like'], '°C')
        st.write('Humidity: ', data['main']['humidity'], '%')
    else:
        # Handle errors
        st.error(f"Error fetching data: {data.get('message', 'Unknown error')}")
