import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from reportlab.pdfgen import canvas as pdf_canvas
from PIL import Image
import os

# Initialize Mediapipe Hands solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75, min_tracking_confidence=0.75)
mp_drawing = mp.solutions.drawing_utils

# Parameters for drawing
drawing_color = (0, 0, 255)  # Default color: red
brush_thickness = 5
eraser_thickness = 50
eraser_mode = False
color_mode = False
whiteboard = None
prev_x, prev_y = 0, 0

# Initialize deque to store points for smoothing the line
points_queue = deque(maxlen=10)

# Webcam capture
cap = cv2.VideoCapture(0)

# Function to draw a color palette on the screen
def draw_color_palette(frame):
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255)]
    for i, color in enumerate(colors):
        cv2.rectangle(frame, (10 + i*60, 10), (60 + i*60, 60), color, -1)
    return colors

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame to create a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB for Mediapipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    result = hands.process(rgb_frame)

    # Initialize whiteboard if not done already
    if whiteboard is None:
        whiteboard = np.ones_like(frame) * 255  # White background

    # Draw color palette on the frame
    colors = draw_color_palette(frame)

    # If a hand is detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Get coordinates of index and middle finger tips
            x1, y1 = int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])
            x2, y2 = int(middle_finger_tip.x * frame.shape[1]), int(middle_finger_tip.y * frame.shape[0])

            # Check if index and middle fingers are close enough (for eraser mode)
            if abs(y1 - y2) < 30:
                eraser_mode = True
            else:
                eraser_mode = False

            # Check if user is selecting a color from the palette
            if 10 < y1 < 60:
                for i, color in enumerate(colors):
                    if 10 + i*60 < x1 < 60 + i*60:
                        drawing_color = color
                        color_mode = True

            if not eraser_mode and not color_mode:
                points_queue.append((x1, y1))

                # Smoothing the line by averaging points in the queue
                if len(points_queue) > 1:
                    smoothed_points = np.mean(points_queue, axis=0).astype(int)
                    x1_smooth, y1_smooth = smoothed_points

                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = x1_smooth, y1_smooth

                    # Draw the line on the whiteboard
                    cv2.line(whiteboard, (prev_x, prev_y), (x1_smooth, y1_smooth), drawing_color, brush_thickness)
                    prev_x, prev_y = x1_smooth, y1_smooth
            elif eraser_mode:
                cv2.circle(whiteboard, (x1, y1), eraser_thickness, (255, 255, 255), -1)
                prev_x, prev_y = 0, 0  # Reset previous coordinates after erasing
            else:
                prev_x, prev_y = 0, 0  # Reset previous coordinates when switching colors
    else:
        prev_x, prev_y = 0, 0  # Reset previous coordinates if no hand is detected

    # Show both the webcam feed and the whiteboard
    cv2.imshow("Air Canvas (Press 's' to Save, 'q' to Quit)", frame)
    cv2.imshow("Whiteboard", whiteboard)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):  # Quit the application
        break

    if key == ord('s'):  # Save the whiteboard as a PDF
        grayscale_whiteboard = cv2.cvtColor(whiteboard, cv2.COLOR_BGR2GRAY)
        grayscale_whiteboard = cv2.bitwise_not(grayscale_whiteboard)

        # Convert OpenCV image to PIL image for saving
        pil_image = Image.fromarray(grayscale_whiteboard)
        image_path = "drawing_output.png"
        pil_image.save(image_path)

        # Generate PDF and insert the saved image
        pdf_filename = "drawing_output.pdf"
        pdf = pdf_canvas.Canvas(pdf_filename)
        pdf.drawImage(image_path, 0, 0, width=600, height=800)
        pdf.save()
        print(f"Drawing saved as {pdf_filename}")

        # Remove the temporary image file
        os.remove(image_path)

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
