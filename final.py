import cv2
import numpy as np
import imutils
import datetime
from inference_sdk import InferenceHTTPClient
from roboflow import Roboflow
import tempfile
from collections import deque

rf = Roboflow(api_key="V2lTkhacO8LY3TPugNVO")

print(rf.workspace())

workspaceId = 'hunter-diminick'
projectId = 'mice-detection-flgeh'

project = rf.workspace(workspaceId).project(projectId)

# Open the camera
cap = cv2.VideoCapture(0)

# Set the resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

firstFrame = None
min_area = 10  # Minimum area size for motion detection

client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="V2lTkhacO8LY3TPugNVO",
)

count = 2
upload_interval = 2  # Adjust the interval for when to upload
frame_count = 0  # Initialize a frame count to track the frames

# Create a deque (double-ended queue) to store the last 20 frames
frame_buffer = deque(maxlen=20)

# Loop over the frames of the video
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Add the current frame to the buffer
    frame_buffer.append(gray)

    # Set the firstFrame to the frame from 20 frames ago if available
    if len(frame_buffer) == 20:
        firstFrame = frame_buffer[0]  # The frame from 20 frames ago

    if firstFrame is None:
        continue

    # Compute the absolute difference between the current frame and first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    # Dilate the thresholded image to fill in holes, then find contours on the thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    text = "No Mouse"

    # Loop over the contours
    for c in cnts:
        # If the contour is too small, ignore it
        if cv2.contourArea(c) < min_area:
            continue

        # Compute the bounding box for the contour, draw it on the frame, and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        text = "Mouse"
        
        # Only upload every 'upload_interval' detections
        # if count == 0:
        with tempfile.NamedTemporaryFile(suffix=".jpg") as temp:
            cv2.imwrite(temp.name, frame)
            # project.upload(image_path=temp.name)
            result = client.run_workflow(
                workspace_name="hunter-diminick",
                workflow_id="mouse-workflow",
                images={
                    "image": temp.name
                }
            )
            print(result)
            print("Frame uploaded "+ datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"))
            
            count = upload_interval

    # Decrease count by one on every detection frame loop
    # if text == "Mouse":
    #     count -= 1

    # Display the resulting frame with motion detection
    cv2.putText(frame, f"Room Status: {text}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    
    # Show the frame
    cv2.imshow("Security Feed", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()