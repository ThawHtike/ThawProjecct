import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from itertools import chain
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, mean_squared_error
from datetime import datetime, timedelta
import pytz

# Load the dataset
data = pd.read_csv('Dataset11-weather-Data.csv')

# Display initial data information
print(data.head())
print(data.shape)
print(data.columns)
print(data.dtypes)
print(data.info())
print(data.Weather.value_counts())
print(data.Weather.unique())
print(data.Weather.nunique())


# Function to create a flattened list from the weather string
def create_list(weather_string):
    list_of_lists = [w.split() for w in weather_string.split(',')]
    return list(chain(*list_of_lists))


# Function to standardize weather data
def get_weather(weather_list):
    if 'Fog' in weather_list and 'Rain' in weather_list:
        return 'RAIN+FOG'
    elif 'Snow' in weather_list and 'Rain' in weather_list:
        return 'SNOW+RAIN'
    elif 'Snow' in weather_list:
        return 'SNOW'
    elif 'Fog' in weather_list:
        return 'FOG'
    elif 'Clear' in weather_list:
        return 'Clear'
    elif 'Cloudy' in weather_list:
        return 'Cloudy'
    else:
        return 'RAIN'


# Create standard weather column
data['Std_Weather'] = data['Weather'].apply(lambda x: get_weather(create_list(x)))

# Sampling and preprocessing
cloudy_df = data[data['Std_Weather'] == 'Cloudy'].sample(600)
clear_df = data[data['Std_Weather'] == 'Clear'].sample(600)
rain_df = data[data['Std_Weather'] == 'RAIN']
snow_df = data[data['Std_Weather'] == 'SNOW']
weather_df = pd.concat([cloudy_df, clear_df, rain_df, snow_df], axis=0)

# Drop unnecessary columns and check for duplicates
weather_df.drop(columns=['Date/Time', 'Weather'], axis=1, inplace=True)

print(weather_df.duplicated().sum())
print(weather_df.isnull().sum())
print(weather_df.describe())

# Correlation matrix and visualizations
cols = ['Temp_C', 'Dew Point Temp_C', 'Rel Hum_%', 'Wind Speed_km/h', 'Visibility_km', 'Press_kPa']
cor_matrix = weather_df[cols].corr()
sns.heatmap(cor_matrix, annot=True)
plt.show()

# Label encoding and feature preparation
label_encoder = LabelEncoder()
weather_df['Std_Weather'] = label_encoder.fit_transform(weather_df['Std_Weather'])
X = weather_df.drop(['Std_Weather'], axis=1)
y = weather_df['Std_Weather']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Classifier': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Gaussian Naive Bayes': GaussianNB()
}

# Train, predict, and evaluate each model
model_accuracies = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    model_accuracies[model_name] = accuracy
    print(f'{model_name} Accuracy: {accuracy:.2f}')

# Convert accuracy results to DataFrame
model_df = pd.DataFrame(model_accuracies.items(), columns=['Model', 'Accuracy'])
print(model_df)

# Cross-validation for Random Forest model
rf_model = RandomForestClassifier()
scores = cross_val_score(rf_model, X_scaled, y, cv=5, scoring='accuracy')
print('Cross-validation scores:', scores)
print('Mean Cross-validation score:', scores.mean())

# Grid search for hyperparameter tuning
parameters = {
    'n_estimators': [50, 100],
    'max_features': ['sqrt', 'log2', None]
}
grid_search = GridSearchCV(estimator=rf_model, param_grid=parameters, cv=3)
grid_search.fit(X_train, y_train)
print('Best Parameters:', grid_search.best_params_)

# Train best model after hyperparameter tuning
best_rf_model = RandomForestClassifier(max_features=grid_search.best_params_['max_features'],
                                       n_estimators=grid_search.best_params_['n_estimators'])
best_rf_model.fit(X_train, y_train)

# Final evaluation of the best model
y_pred_best = best_rf_model.predict(X_test)
best_accuracy = accuracy_score(y_test, y_pred_best)
print(f'Best Random Forest Model Accuracy: {best_accuracy:.2f}')


# Function to get current weather data
def get_current_weather(city):
    api_key = "a5bedda26f30751cafc720db15d72c20"
    url = f'https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}'
    try:
        response = requests.get(url)
        data = response.json()
        return {
            'city': data['name'],
            'current_temp': round(data['main']['temp']),
            'feels_like': round(data['main']['feels_like']),
            'temp_min': round(data['main']['temp_min']),
            'temp_max': round(data['main']['temp_max']),
            'humidity': round(data['main']['humidity']),
            'description': data['weather'][0]['description'],
            'country': data['sys']['country'],
            'wind_gust_dir': data['wind']['deg'],
            'pressure': data['main']['pressure'],
            'WindGustSpeed': data['wind']['speed']
        }
    except Exception as e:
        print(f"Error: {e}")
        return None


# Forecasting future weather
def weather_forecasting():
    city = input('Enter the city name for current weather: ')
    current_weather = get_current_weather(city)

    if current_weather:
        print(f"City: {current_weather['city']}, {current_weather['country']}")
        print(f"Current Temperature: {current_weather['current_temp']}°C")
        print(f"Feels like: {current_weather['feels_like']}°C")
        print(f"Minimum Temperature: {current_weather['temp_min']}°C")
        print(f"Maximum Temperature: {current_weather['temp_max']}°C")
        print(f"Humidity: {current_weather['humidity']}%")
        print(f"Weather Description: {current_weather['description']}")


# Call the weather forecasting function
weather_forecasting()