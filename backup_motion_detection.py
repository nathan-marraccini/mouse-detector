import cv2
import numpy as np

# Open the camera
cap = cv2.VideoCapture(0)

# Set the resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Initialize variables
last_mean = 0
detected_motion = False
frame_rec_count = 0

# Video writer for saving the output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280, 720))

while True:
    ret, frame = cap.read()
    
    # Check if the frame was captured successfully
    if not ret:
        print("Failed to grab frame")
        break
    
    # Display the frame
    cv2.imshow('frame', frame)
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate the mean difference
    result = np.abs(np.mean(gray) - last_mean)
    print(result)
    
    # Update the last mean
    last_mean = np.mean(gray)
    
    # Detect motion
    if result > 5:
        print("Motion detected!")
        print("Started recording.")
        detected_motion = True
    
    # Record frame if motion is detected
    if detected_motion:
        out.write(frame)
        frame_rec_count += 1
    
    # Break the loop on 'q' key press or after recording 240 frames
    if (cv2.waitKey(1) & 0xFF == ord('q')) or frame_rec_count == 240:
        break

# Release the camera and video writer
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
