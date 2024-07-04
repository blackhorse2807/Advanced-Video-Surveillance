import cv2
import numpy as np
import pickle
from shapely.geometry import Point, Polygon

# Load YOLO
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
# layer_names = net.getLayerNames()
# output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
layer_names = net.getLayerNames()

try:
    # Newer OpenCV versions have a different structure
    out_layers = net.getUnconnectedOutLayers().flatten()
except AttributeError:
    # Older OpenCV versions, and different DNN backends can have a different structure
    out_layers = net.getUnconnectedOutLayers()

output_layers = [layer_names[i - 1] for i in out_layers]

# Load the ROI from the pickle file
# with open('clicked_points_intrusion.pkl', 'rb') as f:
#     roi_polygon = pickle.load(f)
with open('clicked_points_intrusion.pkl', 'rb') as f:
    roi_points = pickle.load(f)
# print(roi_points)
roi_points = [x for x in roi_points if x is not None]
# print(roi_points)

roi_polygon = Polygon(roi_points)
# Initialize the video capture
cap = cv2.VideoCapture('input1_.mp4')  # replace with your video file or 0 for webcam

# Function to draw the ROI polygon on the frame
def draw_polygon(frame, roi_polygon):
    cv2.polylines(frame, [np.array(roi_polygon)], True, (255, 0, 0), 3)

# Function to check if a point is inside the ROI polygon
# def check_intrusion(x, y, roi_polygon):
#     point = np.array([x, y])
#     inside = cv2.pointPolygonTest(np.array(roi_polygon), tuple(point), False)
#     return inside >= 0
def is_point_in_roi(point, polygon):
    return polygon.contains(Point(point))

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video or can't read the frame

    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes and class_ids[i] == 0:
            x, y, w, h = boxes[i]
            center_point = (x + w // 2, y + h // 2)
            if is_point_in_roi(center_point, roi_polygon):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.circle(frame, center_point, 5, (0, 255, 0), -1)
                cv2.putText(frame, 'Intruder detected!', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                # print("Intruder detected in the region of interest!")

    cv2.imshow("Frame", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




#can add count functionality
