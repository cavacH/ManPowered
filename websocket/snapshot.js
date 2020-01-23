'use strict';
$('.ui.dropdown').dropdown();
function popup_modal() {
    $('#add-camera').modal('show');
    $('#video-select').removeClass('active visible');
}
function hide_modal() {
    $('#add-camera').modal('hide');
}

var videoSelect = document.querySelector('#video-select');
var imageDisplay = document.querySelector('#image-display');
var canvas = document.querySelector('canvas');
var video = document.querySelector('video');
var fps = document.querySelector('#fps');
navigator.mediaDevices.enumerateDevices().then(gotDevices).catch(handleError);
var localStream;
var imageCapture;

var ws = new WebSocket('ws://localhost:8000/');
var date = new Date();
var mill = date.getTime();

ws.onmessage = (e) => {
    snapshot();
    // console.log(e.data);
    var current = (new Date()).getTime();
    fps.innerHTML = 1000. / (current - mill);
    mill = current;
    imageDisplay.src = e.data;
    return false;
};

var width = 640;
var height = 480;

// et media stream
function add_video() {
    var constraints = {
        video: {
            deviceId: {
                exact: videoSelect.value
            },
            width: width,
            height: height 
        }
    };
    navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
        video.srcObject = stream;
        localStream = stream;
        snapshot();
        // var snapshotInterval = setInterval(snapshot, 100);
        // imageCapture = new ImageCapture(stream.getVideoTracks()[0]);
        // console.log(imageCapture);
    });
}

function snapshot() {
    var context = canvas.getContext('2d');
    canvas.height = height;
    canvas.width = width;
    context.drawImage(video, 0, 0, width, height);
    var data = canvas.toDataURL('image/jpeg');
    ws.send(data);
    // console.log(data);
    // imageCapture.takePhoto().then(blob => {
    //     console.log('Photo taken: ' + blob.type + ', ' + blob.size + 'B');
    //     imageDisplay.src = URL.createObjectURL(blob);
    // })
    // ws.send('vancior');
}

function gotDevices(deviceInfos) {
    for (var i = 0; i !== deviceInfos.length; ++i) {
        var deviceInfo = deviceInfos[i];
        var option = document.createElement('option');
        option.value = deviceInfo.deviceId;
        if (deviceInfo.kind === 'videoinput') {
            option.text = deviceInfo.label || 'camera ' + (videoSelect.length);
            videoSelect.appendChild(option);
            console.log('Found camera: ', deviceInfo);
        }
        else {
            // console.log('Found on other kind of source/device: ', deviceInfo);
        }
    }
}

function handleError(evt) {
    console.log(evt)
}
