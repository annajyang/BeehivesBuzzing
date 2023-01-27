# Smart Bee Colony Monitor: Internet of Things Device & Fusion Convolutional Neural Network for Queen Assessment

## Hardware
All of the code for the IoT device was run through Visual Studio Code. There are two main software folders: “server” and “i2s_sampling.”  
  
The “i2s_sampling” folder holds the code that is loaded onto the ESP32 and controls the data collection. The server IP is configured with the Wi-Fi name and password, and the software can also be updated for changes in data collection frequency and the sound sample length.
  
The “server” folder contains the program that the server runs on. Additional data was also pulled from OpenWeatherMap API, which acquired ambient humidity, pressure, cloudiness, precipitation, temperature, wind speed, and gust speed based on the location of the device. Once the server is started, the ESP32 can be plugged into any USB power source to begin data collection immediately.  
  
The code for the device was largely derived from [atomic14's code on GitHub](https://github.com/atomic14/esp32_audio), with changes to accomodate the needs of the project.

## Data Processing
Most of the data processing code was written by myself, including: compiling the data into a csv, labeling each data sample, converting the .raw files to .wav files,  

To segment the audio files into 60 second intervals and convert the samples into MFCCs, I used [inesnola's code on GitHub](https://github.com/inesnolas/Audio_based_identification_beehive_states), which was published as part of the paper “Audio-based beehive state recognition” (Nolasco, 2018). This code is located in the folder "Beehive_state_classification," in the files "utilsBeehiveState.py, "data_processing.py," "tests_.py," and "data_processing_beeNotbee.py." The code in these files were edited by me to fit the necessities of my project.  

The rest of the code in that folder is my own, with segments taken from Nolasco's work to split the data into test, train, and val sets by ratios.

## Machine Learning
