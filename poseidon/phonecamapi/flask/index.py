from flask import Flask, request, Response, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/videostream', methods=['POST'])
def handle_video_stream():
    video_chunks = request.files.getlist('videoChunk')
    # Handle the incoming video data here
    for chunk in video_chunks:
        print(chunk)
        # Process each video chunk
        # ...
    return 'Video stream received successfully!'

# @app.route('/api/getvideostream')
# def send_video_stream():
#     # Generate the video stream here
#     video_data = generate_video_stream()

#     # Create a Flask Response object with the video data
#     return Response(video_data, mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/getvideostream', methods=['GET'])
def send_videostream():
    

# # Helper function to generate video stream
# def generate_video_stream():
#     while True:
#         # Generate the next frame of the video stream
#         frame = generate_next_frame()

#         # Convert the frame to bytes
#         frame_bytes = frame.tobytes()

#         # Yield the frame as part of a multipart response
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# # Helper function to generate the next frame of the video stream
# def generate_next_frame():
#     # Generate the next frame of the video stream here
#     # ...
#     return frame


if __name__ == '__main__':
    app.run(host='localhost', port=8080)
