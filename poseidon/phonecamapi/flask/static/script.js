const video = document.getElementById("video");
const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");

let mediaRecorder;
let recordedChunks = [];

startButton.onclick = async function startRecording() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
    });
    mediaRecorder = new MediaRecorder(stream);
    const videoElement = document.getElementById("video");
    videoElement.srcObject = stream;
    videoElement.play();
    document.body.appendChild(videoElement);

    mediaRecorder.ondataavailable = function (event) {
        if (event.data.size > 0) {
            recordedChunks.push(event.data);
            sendChunks();
        }
    };

    mediaRecorder.start();

    stopButton.addEventListener("click", () => {
        mediaRecorder.stop();
        sendChunks();
    });
};

async function sendChunks() {
    if (recordedChunks.length > 0) {
        const formData = new FormData();
        for (let i = 0; i < recordedChunks.length; i++) {
            formData.append(
                "videoChunk",
                recordedChunks[i],
                "recorded-video.webm"
            );
        }
        recordedChunks = [];

        const response = await fetch("http://localhost:8080/api/videostream", {
            method: "POST",
            body: formData,
        });

        console.log(response);
    }
}
