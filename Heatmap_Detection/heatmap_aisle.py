import cv2
import numpy as np
import pickle
#Inferencing a video
video_path = "input1_.mp4"
# video_path = 
print(video_path)
cap = cv2.VideoCapture(video_path)
while (cap.isOpened()):
    #Capture frame-by-frame
    ret, frame = cap.read()
    (frame_height, frame_width) = frame.shape[:2]
    print("frame_height, frame_width:",frame_height, frame_width)
    break
image = frame.copy()
#image = cv2.resize(image,(800,600))
# List to store the clicked points
clicked_points = []
# Flag to indicate whether to draw lines or not
draw_lines = False

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global draw_lines

    if event == cv2.EVENT_LBUTTONDOWN:  # Left button down event
        clicked_points.append((x, y))
        print(f"Clicked point: ({x}, {y})")
        draw_lines = True

# Create a named window and set the mouse callback
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)


while True:

    cv2.imshow("Image", image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("c"):
        clicked_points.append(None)
        draw_lines = False

    elif key == ord("e"):
        break

    if draw_lines and len(clicked_points) > 1:
        for i in range(len(clicked_points) - 1):
            pt1 = clicked_points[i]
            pt2 = clicked_points[i + 1]
            if pt1 and pt2:
                cv2.line(image, pt1, pt2, (255, 255, 0), 2)

# Display all the clicked points
print("Clicked points:")
print("Section 1:")
a = len(clicked_points)
t = 2
for i in range(a):
    if clicked_points[i] is None:
        print("\nSection", t)
        t += 1
    else:
        print(clicked_points[i][0], "\t", clicked_points[i][1])

print(clicked_points)
# Specify the desired file path for the pickle file
pickle_file_path = 'clicked_points.pkl'

# Open the file in binary mode and write the array using pickle.dump()
with open(pickle_file_path, 'wb') as file:
    pickle.dump(clicked_points, file)

print(f"Array saved to {pickle_file_path}")

# Release the OpenCV windows
cv2.destroyAllWindows()