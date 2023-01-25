# Smart Bee Colony Monitor: Internet of Things Device & Fusion Convolutional Neural Network for Queen Assessment

## Hardware
All of the code for the IoT device was run through Visual Studio Code. There are two main software folders: “server” and “i2s_sampling.”  
  
The “i2s_sampling” folder holds the code that is loaded onto the ESP32 and controls the data collection. The server IP is configured with the Wi-Fi name and password, and the software can also be updated for changes in data collection frequency and the sound sample length.
  
The “server” folder contains the program that the server runs on. Additional data was also pulled from OpenWeatherMap API, which acquired ambient humidity, pressure, cloudiness, precipitation, temperature, wind speed, and gust speed based on the location of the device. Once the server is started, the ESP32 can be plugged into any USB power source to begin data collection immediately.  
  
The code for the device was largely derived from atomic14's code on GitHub (https://github.com/atomic14/esp32_audio), with changes to accomodate the needs of the project.

## Data Processing


## Machine Learning
