import cv2
import face_recognition
import mysql.connector

# initialize the video stream
cap = cv2.VideoCapture(0)

# load the facial recognition model
known_face_encodings = []
known_face_names = []
# TODO: Load your known face encodings and names here.

# Connect to the mysql database
mydb = mysql.connector.connect(
    host="localhost",
    user="username",
    password="password",
    database="database_name"
)

# Initialize the cursor object
mycursor = mydb.cursor()

# Start the video stream
while True:
    # Capture the frames
    ret, frame = cap.read()

    # Detect faces using OpenCV's built-in face detector
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face found in the current frame
    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Recognize the face
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        # Search for the face in the database
        if True in matches:
            # TODO: Alert if face found in the database.
            pass
        else:
            # Store the face in the database
            known_face_encodings.append(face_encoding)
            # TODO: Get the name of the person in the face encoding
            known_face_names.append(name)
            sql = "INSERT INTO faces (name, encoding) VALUES (%s, %s)"
            val = (name, face_encoding.tostring())
            mycursor.execute(sql, val)
            mydb.commit()

    # Display the frame with face detections
    cv2.imshow('frame', frame)

    # Stop the loop by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()
