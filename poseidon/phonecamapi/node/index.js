const express = require("express");
const multer = require("multer");
const fs = require("fs");

const app = express();
const upload = multer({ dest: "uploads/" });
const PORT = 8080;

app.use(express.static(__dirname + "/public"));
app.use(express.raw({ type: 'video/webm', limit: '50mb' }));

app.get("/", (request, response) => {
    fs.readFile("./index.html", "utf8", (err, html) => {
        if (err) {
            response.status(500).send("not working");
        }
        response.send(html);
    });
});

app.get("/index.html", (request, response) => {
    fs.readFile("./index.html", "utf8", (err, html) => {
        if (err) {
            response.status(500).send("not working");
        }
        response.send(html);
    });
});

app.get("/script.js", (request, response) => {
    fs.readFile("./script.js", "utf8", (err, js) => {
        if (err) {
            response.status(500).send("not working");
        }
        response.set("Content-Type", "application/javascript");
        response.send(js);
    });
});

var videoFile;
app.post("/api/videostream", (req, res) => {
    // Handle the incoming video data here
    videoFile = req.body;
    console.log(videoFile)
    console.log(req.files);
    res.status(200).send("Video stream received successfully!");
});

app.get("/api/videostream", (req, res) => {
    
    res.status(200).send(videoFile);
});

app.listen(PORT, () => console.log(`Running on http://localhost:${PORT}`));
