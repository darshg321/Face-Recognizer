const express = require("express");
const multer = require("multer");
const fs = require('fs');

const app = express();
const upload = multer({ dest: "uploads/" });
const PORT = 8080;

app.use(express.static(__dirname + "/public"));

app.get("/", (request, response) => {
    fs.readFile("./index.html", "utf8", (err, html) => {
        if (err) {
            response.status(500).send("not working")
        }
        response.send(html)
    })
})

app.get("/index.html", (request, response) => {
    fs.readFile("./index.html", "utf8", (err, html) => {
        if (err) {
            response.status(500).send("not working")
        }
        response.send(html)
    })
})

app.get("/script.js", (request, response) => {
    fs.readFile("./script.js", "utf8", (err, js) => {
        if (err) {
            response.status(500).send("not working")
        }
        response.set("Content-Type", "application/javascript")
        response.send(js)
    })
})

app.post("/api/videostream", upload.single("video"), (req, res) => {
    // Handle video file here
    console.log(req.file);

    fs.rename(`uploads/${req.file.filename}`, 'uploads/video.webm', (err) => {
        if (err) throw err;
        console.log('Rename complete!');
      });

    res.status(200).send("Video uploaded successfully");
});



app.listen(PORT, () => console.log(`Running on http://localhost:${PORT}`))