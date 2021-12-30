import express from 'express';
import {appendFile} from 'fs'

const port = 5003; // port to listen

let time1 = 'date and time';
let fileTime1 = 'date and time';
let time2 = 'date and time';
let fileTime2 = 'date and time';
let time3 = 'date and time';
let fileTime3 = 'date and time';

const app = express();
var fs = require('fs');
let time = 'date and time';
let fileTime = 'date and time';

let getTime = function () {
  const date_ob = new Date();
  var date = ("0" + date_ob.getDate()).slice(-2);
  var month = ("0" + (date_ob.getMonth() + 1)).slice(-2);
  var year = date_ob.getFullYear();
  var hours = ("0" + date_ob.getHours()).slice(-2);
  var minutes = ("0" + date_ob.getMinutes()).slice(-2);
  var seconds = ("0" + date_ob.getSeconds()).slice(-2);

  time = year + "-" + month + "-" + date + " " + hours + ":" + minutes + ":" + seconds;
  fileTime = year + "-" + month + "-" + date + "--" + hours + "-" + minutes + "-" + seconds;

  //console.log(time);
};

app.use(express.text());
app.use(express.raw());

// receive sensor data and other stuff
app.post('/bme_samples1', function (req, res) {
  getTime();
  time1 = time;
  fileTime1 = fileTime;

  console.log("Got sensor data from 1");

  if (!fs.existsSync('./data_1')){
    fs.mkdirSync('./data_1');
  }

  appendFile('./data_1/data_1.txt', time1 + "\t" + req.body + "\t" + fileTime1 + "_1.raw\n", () => {
    res.send('OK');
  });
});

app.post('/bme_samples2', function (req, res) {
  getTime();
  time2 = time;
  fileTime2 = fileTime;
  console.log("Got sensor data from 2");

  if (!fs.existsSync('./data_2')){
    fs.mkdirSync('./data_2');
  }

  appendFile('./data_2/data_2.txt', time2 + "\t" + req.body + "\t" + fileTime2 + "_2.raw\n", () => {
    res.send('OK');
  });
});

app.post('/bme_samples3', function (req, res) {
  getTime();
  time3 = time;
  fileTime3 = fileTime;
  console.log("Got sensor data from 3");
  
  if (!fs.existsSync('./data_3')){
    fs.mkdirSync('./data_3');
  }

  appendFile('./data_3/data_3.txt', time3 + "\t" + req.body + "\t" + fileTime3 + "_3.raw\n", () => {
    res.send('OK');
  });
});

// receive data from microphone
app.post('/i2s_samples1', function (req, res) {
  console.log(`Got ${req.body.length} I2S bytes from ` + 1);
  appendFile('./data_1/' + fileTime1 + '_1.raw', req.body, () => {
    res.send('OK');
  });
});

app.post('/i2s_samples2', function (req, res) {
  console.log(`Got ${req.body.length} I2S bytes from ` + 2);
  appendFile('./data_2/' + fileTime2 + '_2.raw', req.body, () => {
    res.send('OK');
  });
});

app.post('/i2s_samples3', function (req, res) {
  console.log(`Got ${req.body.length} I2S bytes from ` + 3);
  appendFile('./data_3/' + fileTime3 + '_3.raw', req.body, () => {
    res.send('OK');
  });
});

// start express server
app.listen(port, '0.0.0.0', () => {
  console.log(`server started at http://0.0.0.0:${port}`);
});