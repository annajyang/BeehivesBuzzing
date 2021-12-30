#include <Arduino.h>
#include <WiFi.h>
#include <Wire.h>
#include <HTTPClient.h>
#include <Adafruit_BME280.h>
#include <Adafruit_Sensor.h>
#include <Arduino_Json.h>
#include "WiFiCredentials.h"
#include "I2SMEMSSampler.h"

RTC_DATA_ATTR int bootCount = 0;
int i2sSendCounter = 0; // 'global' variable bad. don't do this
I2SSampler *i2sSampler = NULL;
Adafruit_BME280 bme; 

String httpWeatherGetRequest(const char*);
String formatWeatherData();
void sendDataMIC(uint8_t*, size_t);
void i2sMemsWriterTask(void*);
void printWakeupReason();
void printLocalTime();
void bmeInitialize();
void i2sRecordTask();
void deepSleepTask();
void bmePrintData();
void sendDataBME();
void WiFiSetup();
void HTTPSetup();

// i2s config for reading from i2s micorphone
i2s_config_t i2sMemsConfigBothChannels = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = 16000,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_I2S),
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 20,
    .dma_buf_len = 1024,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0};

// i2s microphone pins
i2s_pin_config_t i2sPins = {
    .bck_io_num = GPIO_NUM_32,
    .ws_io_num = GPIO_NUM_25,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = GPIO_NUM_33};

// setup - everything runs here
void setup() {
  Serial.begin(115200); // set baud rate

  bmeInitialize();    // initialize sensors

  Serial.println("\nDevice number: " + String(DEVICE_NUMBER));  // print device number

	Serial.println("\nBoot number: " + String(++bootCount));  // print start number

	printWakeupReason(); 	// print the wakeup reason for ESP32

  WiFiSetup();  // launch WiFi

  // printLocalTime(); // print time

  // bmePrintData(); //print BME readings

  HTTPSetup();  // setup the HTTP Client

  sendDataBME();  // send bme data to server

  i2sRecordTask();  // start sampling from INMP441

  deepSleepTask();  // go to sleep
}

// everything taken care of by tasks
void loop() {
  // no-op
}

// initialize sensors and LED
void bmeInitialize() {
  bool status;
  pinMode(2, OUTPUT);
  status = bme.begin(0x76); 
  bme.setSampling();
  if (!status) {
    Serial.println("COULD NOT FIND A VALID BME280 SENSOR!");
  }
}

// launch WiFi
void WiFiSetup() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(SSID, PASSWORD);
  Serial.println("\nConnecting to WiFi...");
  if (WiFi.waitForConnectResult() != WL_CONNECTED) {
    Serial.println("Connection Failed! Rebooting...");
    delay(5000);
    ESP.restart();
  }
  Serial.printf("Connected to %s\n\n", SSID );
}

// starts the HTTP clients
void HTTPSetup() {
  wifiClientMIC = new WiFiClient();
  httpClientMIC = new HTTPClient();
  wifiClientBME = new WiFiClient();
  httpClientBME = new HTTPClient();
}

// send bme sensor data to server
void sendDataBME() {
  Serial.println("Sensor read task started");
  String combineData = String(bme.readTemperature()) + "\t" + bme.readHumidity() 
    + "\t" + (bme.readPressure() / 100.0F) + "\t" + formatWeatherData();
  digitalWrite(2, HIGH);
  httpClientBME->begin("http://" + String(SERVER_IP) + ":" + SERVER_PORT + "/bme_samples" + String(DEVICE_NUMBER));
  httpClientBME->addHeader("content-type", "text/plain");
  httpClientBME->POST(String(combineData));
  httpClientBME->end();
  digitalWrite(2, LOW);
}

// get weather data from openWeatherMap API
String httpWeatherGetRequest(const char* serverName) {
  WiFiClient wifiClientWeather;
  HTTPClient httpClientWeather;

  httpClientWeather.begin(wifiClientWeather, serverName);
  int httpResponseCode = httpClientWeather.GET();
  String payload = "{}"; 

  if (httpResponseCode > 0) {
    payload = httpClientWeather.getString();
  }
  else {
    Serial.print("Error code: ");
    Serial.println(httpResponseCode);
  }
  httpClientWeather.end();
  return payload;
}

// format weather data
String formatWeatherData() {
  int humidity, pressure, cloudiness, conditionID, rain;
  double temperature, windSpeed, gustSpeed, lat, lon;

  String jsonBuffer = httpWeatherGetRequest(serverPath.c_str());
  //Serial.println(jsonBuffer);   //debugging
  JSONVar weatherObject = JSON.parse(jsonBuffer);
  if (JSON.typeof(weatherObject) == "undefined") {
    Serial.println("Parsing input failed!");
  }

  temperature = weatherObject["main"]["temp"];
  temperature -= 273.15;  
  humidity = weatherObject["main"]["humidity"];
  pressure = weatherObject["main"]["pressure"];
  windSpeed = weatherObject["wind"]["speed"];
  gustSpeed = weatherObject["wind"]["gust"];
  conditionID = weatherObject["weather"][0]["id"];
  cloudiness = weatherObject["clouds"]["all"];
  rain = weatherObject["rain"]["1hr"];
  lat = weatherObject["coord"]["lat"];
  lon = weatherObject["coord"]["lon"];

  return String(temperature) + "\t" + humidity + "\t" + pressure 
    + "\t" + windSpeed + "\t" + gustSpeed + "\t" + conditionID 
    + "\t" + cloudiness + "\t" + rain + "\t" + lat + "\t" + lon;
}

// inmp441 i2s recording task
void i2sRecordTask() {
  i2sSampler = new I2SMEMSSampler(i2sPins);  // direct i2s input from INMP441
  TaskHandle_t i2sMemsWriterTaskHandle;  // set up the i2s sample writer task
  xTaskCreatePinnedToCore(i2sMemsWriterTask, "I2S Writer Task", 4096, i2sSampler, 1, &i2sMemsWriterTaskHandle, 1);
  i2sSampler->start(I2S_NUM_1, i2sMemsConfigBothChannels, 49152, i2sMemsWriterTaskHandle);  // start sampling
  Serial.println("i2sReaderTask started");
}

// task to write samples to our server
void i2sMemsWriterTask(void *param) {
  I2SSampler *sampler = (I2SSampler *)param;
  const TickType_t xMaxBlockTime = pdMS_TO_TICKS(100);
  while (true) {
    uint32_t ulNotificationValue = ulTaskNotifyTake(pdTRUE, xMaxBlockTime);   // wait for some samples to save
    if (ulNotificationValue > 0) {
      sendDataMIC((uint8_t *)sampler->getCapturedAudioBuffer(), sampler->getBufferSizeInBytes());
    }
  }
}

// microphone data sending function
void sendDataMIC(uint8_t *bytes, size_t count) {
  digitalWrite(2, HIGH);
  httpClientMIC->begin("http://" + String(SERVER_IP) + ":" + SERVER_PORT + "/i2s_samples" + String(DEVICE_NUMBER));
  httpClientMIC->addHeader("content-type", "application/octet-stream");
  httpClientMIC->POST(bytes, count);
  httpClientMIC->end();
  i2sSendCounter++;
  digitalWrite(2, LOW);
}

// puts esp32 in deep sleep power mode
void deepSleepTask() {
  double TIME_TO_SLEEP = COLLECT_INTERVAL - SEND_REPETITIONS * 1.73;  // check all data has been sent out
  while(i2sSendCounter < SEND_REPETITIONS);
	esp_sleep_enable_timer_wakeup(TIME_TO_SLEEP * 1000000); // set sleep timer
	Serial.println("Setup ESP32 to sleep for " + String(int(TIME_TO_SLEEP)) +
	" Seconds");
	esp_deep_sleep_start(); // start sleep
}

// debugging - prints wakeup reason in serial monitor
void printWakeupReason() {
  esp_sleep_wakeup_cause_t wakeup_reason;
  wakeup_reason = esp_sleep_get_wakeup_cause();
  switch(wakeup_reason) {
    case 1  : Serial.println("Wakeup caused by external signal using RTC_IO"); break;
    case 2  : Serial.println("Wakeup caused by external signal using RTC_CNTL"); break;
	  case 3  : Serial.println("Wakeup caused by touchpad"); break;
    case 4  : Serial.println("Wakeup caused by timer"); break;
    case 5  : Serial.println("Wakeup caused by ULP program"); break;
    default : Serial.println("Wakeup was not caused by deep sleep"); break;
  }
}

// debugging - prints current time from ntp server
void printLocalTime() {
  configTime(GMT_OFFSET_SEC, DAYLIGHT_OFFSET_SEC, NTP_SERVER_URL);
  struct tm timeinfo;
  if(!getLocalTime(&timeinfo))
    Serial.println("Failed to obtain time");
  Serial.println(&timeinfo, "Time: %c");
}

// debugging - prints BME data 
void bmePrintData() {
  Serial.print("\nTemperature = ");
  Serial.print(bme.readTemperature());
  Serial.println(" *C");
  
  Serial.print("Pressure = ");
  Serial.print(bme.readPressure() / 100.0F);
  Serial.println(" hPa");

  Serial.print("Approx. Altitude = ");
  Serial.print(bme.readAltitude(SEALEVELPRESSURE_HPA));
  Serial.println(" m");

  Serial.print("Humidity = ");
  Serial.print(bme.readHumidity());
  Serial.println(" %\n");
}