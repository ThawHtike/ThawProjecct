import requests

city="Yangon"

api_key = "a5bedda26f30751cafc720db15d72c20"

url = f'https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric'

response =requests.get(url)

data = response.json()

print(data)

print('location longitude is ',data['coord']['lon'])
print('location latitude is ',data['coord']['lat'])
print('weather is ',data['weather'][0]['description'])
print('current temperature is ',data['main']['temp'])

print('current minimum temperature is ',data['main']['temp_min'])

print('current maximum temperature is ',data['main']['temp_max'])


print('current temperature feel like is ',data['main']['feels_like'])
print('current temperature humidity is ',data['main']['humidity'])









print('current temperature feel like is ',data['main']['feels_like'])

print('current temperature humidity is ',data['main']['humidity'])
