#define SSID "Xiaomi_81C6"
#define PASSWORD "yangzhiping"

#define DEVICE_NUMBER "1"
#define SEND_REPETITIONS 20  // 1 send cycle is about 1.536 seconds and 49152 bytes
#define COLLECT_INTERVAL  60   // record interval (in seconds)
#define SEALEVELPRESSURE_HPA 1013.25  //  seapressure - change this once in a while

#define SERVER_IP "192.168.86.55" // server ipv4 address
#define SERVER_PORT "5003"  // server port

#define WEATHER_API_KEY "ae431dab50c412ffc469c7f29b4dc90a"
String city = "Campbell";
String state = "CA";
String countryCode = "US";
String serverPath = "http://api.openweathermap.org/data/2.5/weather?q=" + city + "," + state + "," + countryCode + "&APPID=" + WEATHER_API_KEY;

#define NTP_SERVER_URL "pool.ntp.org"
#define GMT_OFFSET_SEC -28800     // timezone
#define DAYLIGHT_OFFSET_SEC 3600  // daylight savings offset

// idk
WiFiClient *wifiClientMIC = NULL;
HTTPClient *httpClientMIC = NULL;
WiFiClient *wifiClientBME = NULL;
HTTPClient *httpClientBME = NULL;