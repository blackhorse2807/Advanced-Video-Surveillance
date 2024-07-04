import numpy as np
import time
import cv2
import math
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import pickle

labelPath = "obj.names"
LABELS = []
# print(labelPath)

f = open(labelPath, 'r')
data = f.readlines()

for d in data:
    LABELS.append(d[:-1])

#checking if a point lies inside a polygon(or aisle)
def point_in_polygon(point, polygon):
    point = Point(point)
    polygon = Polygon(polygon)
    return polygon.contains(point)

# Specify the file path of the pickle file containing a list of points of aisles. Each aisle is separated by None value
pickle_file_path = "clicked_points.pkl"

# Open the file in binary mode and load the array using pickle.load()
with open(pickle_file_path, 'rb') as file:
    polygon_points = pickle.load(file)
coordinates_len = len(polygon_points)
a = polygon_points.count(None)  # number of None values. Aisles are separated by None values
section_count = np.zeros((a + 1, 1),dtype='int')  # This will store the number of points in a particular section
# print(polygon_points)
weightsPath = "yolov3-tiny.weights"
configPath = "yolov3-tiny.cfg"
# load our YOLO object detector
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Inferencing a video
# video_path = "input1_.mp4"
#   #path to video file
video_path = 'input1_.mp4'
# print(video_path)
cap = cv2.VideoCapture(video_path)

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# print(length)

first_init = True
max_p_cnt = 0
tot_p_cnt = 0
avg_p_cnt = 0
first_iteration_indicator = 1
frame_cnt = 0

while (cap.isOpened()):

    # Capture frame-by-frame
    ret, image = cap.read()
    #image =  cv2.resize(image,(800,600))
    (frame_height, frame_width) = image.shape[:2]  # dimensions of video frame
    # print("frame_height, frame_width:",frame_height, frame_width)
    if ret == True:
        blank = np.zeros((frame_height, frame_width), dtype= 'int') # an array of image size used to store some values at the coordinates where object is detected in a video frame
        break

cap = cv2.VideoCapture(video_path)
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, image = cap.read()
    if ret == True:
        #image = cv2.resize(image, (800, 600))
        frame_cnt += 1
        # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
        # construct a blob from the input image and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes and
        # associated probabilities
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (640, 640), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()
        # show timing information on YOLO
        # print("[INFO] YOLO took {:.6f} seconds".format(end - start))
        # initialize our lists of detected bounding boxes, confidences, and
        # class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # filter out weak predictions by ensuring the detected probability is greater than the minimum probability
                if confidence > 0.2:   # detection at 20% confidence
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([frame_width, frame_height, frame_width, frame_height])
                    (centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top
                    # top left corner of the bounding box
                    topX = int(centerX - (width // 2))
                    topY = int(centerY - (height // 2))
                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([topX, topY, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        print(boxes)
        print(confidences)
        print(classIDs)
        # apply non-maxima suppression to suppress weak, overlapping bounding boxes
        idx = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.25)
        print(idx)
        print(len(idx))
        # ensure at least one detection exists
        if len(idx) > 0:
            # loop over the indexes we are keeping
            for i in idx.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1]) # top left corner of bounding box
                (w, h) = (boxes[i][2], boxes[i][3]) # width height of bounding box
                #coordinates of foot of person:
                footX = x + w // 2
                footY = y + h
                print('classids',classIDs[i])
                print('labels_classids', LABELS[classIDs[i]])
                if LABELS[classIDs[i]] == "person":
                    # draw a bounding box rectangle and label on the image
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    cv2.putText(image, str(int(confidences[i] * 100)) + "%", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 0, 0), 1, cv2.LINE_AA)

                    none_encounter = 0
                    polygon = []
                    point = (footX,footY)

                    #calculating time in aisle
                    for i in range(coordinates_len): # iterating through all the values in the array containing coordinates of aisles
                        if polygon_points[i] is not None:  #checking if the value is not a None value
                            polygon.append(polygon_points[i])  # appending coordinates of a aisle in empty polygon array
                            if i == coordinates_len - 1 or polygon_points[i + 1] is None:  # checking if are at the end of list or at the last coordinate of a polygon
                                    check = point_in_polygon(point, polygon) #checking if a point lies inside an aisle
                                    if check:
                                        section_count[none_encounter] += 1 #if point is detected then number of points in a section is incremented
                                    polygon = []
                        else: #if none value is encountered that is next coordinate will be a point of next aisle
                            none_encounter+=1
                            polygon=[]

                    polygon = []
                    #heatmap in aisle
                    for i in range(coordinates_len):
                        if polygon_points[i] is not None:
                            polygon.append(polygon_points[i])
                            if (i == coordinates_len - 1 or polygon_points[i + 1] is None):
                                check = point_in_polygon(point, polygon)
                                if check:
                                    # iterating within a circle of radius w/6 with center foot of object
                                    for p in range(-(w // 6),w // 6 + 1):
                                        for q in range(-(w // 6),(w // 6 + 1)):
                                            a = footX + p  # a is x coordi of neighbor point of (footX,footY) which is at distance p along the x axis
                                            b = footY + q  # b is y coordi of a neighbor point of (footX,footY) which is q distance above the foot
                                            distance = math.sqrt((a - footX) ** 2 + (b - footY) ** 2)
                                            if (frame_width - 3 < a):  # if concerned point comes to the end of the frame we just ignore it
                                                continue
                                            if (frame_height - 3 < b):
                                                continue
                                            if(point_in_polygon((a,b),polygon) == False): # if a point lies outside our concerned region we just ignore it
                                                continue
                                            if ((distance) <=(w // 6)):
                                                blank[b, a] = blank[b, a] + 1  # the value of blank array at the point where object is detected and its neighborhood is increased
                                            # checking the section in which the object lies
                                    polygon = []
                                else:
                                    none_encounter+=1
                                    polygon=[]
        cv2.imshow("Person count", image)
        cv2.waitKey(1)
        # if(frame_cnt==100):
            # break
    else:
        cap.release()
        cv2.destroyAllWindows()
        break

max_val = np.max(blank)


def interpolate(i): # color scheme for the 99% of data recorded for person detection
    if i==9:
        R = 227
        G = 35
        B = 27
    elif i==8:
        R = 228
        G = 58
        B = 28
    elif i==7:
        R = 231
        G = 96
        B = 31
    elif i == 6:
        R = 236
        G = 137
        B = 35
    elif i==5:
        R = 241
        G = 178
        B = 39
    elif i==4:
        R = 248
        G = 222
        B = 44
    elif i==3:
        R = 245
        G = 253
        B = 47
    elif i==2:
        R = 210
        G = 252
        B = 44
    elif i==1:
        R = 176
        G = 251
        B = 42
    elif i==0:
        R = 146
        G = 250
        B = 40
    color = [B, G, R]
    return color

def interpolate2(i): # color scheme for the minimum 1% of data recorded for person detection
    if i==9:
        R = 120
        G = 250
        B = 38
    elif i==8:
        R = 103
        G = 249
        B = 37
    elif i==7:
        R = 100
        G = 249
        B = 44
    elif i == 6:
        R = 100
        G = 249
        B = 77
    elif i==5:
        R = 100
        G = 250
        B = 116
    elif i==4:
        R = 101
        G = 250
        B = 157
    elif i==3:
        R = 101
        G = 250
        B = 200
    elif i==2:
        R = 101
        G = 251
        B = 242
    elif i==1:
        R = 89
        G = 219
        B = 251
    elif i==0:
        R = 71
        G = 175
        B = 250
    color = [B, G, R]
    return color

cap = cv2.VideoCapture(video_path)
while True:
    ret, frame = cap.read()
    #frame =  cv2.resize(frame,(800,600))
    break

#frame = cv2.imread("final.png") #manually reading an image from the video where no object was present to make heatmap
#frame = cv2.resize(frame, (frame_width,frame_height)) # resizing the image according to the size of video frame
overlay = frame.copy() # a duplicate image of the frame will used in creating heatmaps
height,width,channel = overlay.shape
print("overlay dimensions", height,width,channel)
# print("hello")# random print statement to check till where program is executed
def remove_less_than_percent(arr): # function to separate minimum 1 percent values of the data
    max_val = arr[-1]  # Last element of the sorted array is the maximum value
    threshold = max_val * 0.01  # 1 percent of the maximum value

    # Find the first element greater than the threshold
    index = 0
    while index < len(arr) and arr[index] <= threshold:
        index += 1

    # Remove elements before the threshold
    if index > 0:
        arr2 = arr[:index]
        arr = arr[index:]

    return arr, arr2
maximum_value_in_data = np.max(blank)
array_1d = blank.flatten() # creating a 1d array of the 2d data
sorted_array = np.sort(array_1d) #sorting the data in ascending order
#sorted_array contain maximum 99% values , small_array contain minimum 1% values
sorted_array,small_values = remove_less_than_percent(sorted_array)



def create_clusters(arr):
    # Remove duplicate points from the input array
    unique_arr = np.unique(arr)

    # Reshape the array into a 2D format expected by K-means algorithm
    X = unique_arr.reshape(-1, 1)

    # Determine the number of clusters based on the number of unique samples
    num_clusters = min(10, len(unique_arr))

    if num_clusters == 0:
        return [], [], [], 0

    # Create a K-means object with the determined number of clusters
    kmeans = KMeans(n_clusters=num_clusters)

    # Fit the data to the K-means algorithm
    kmeans.fit(X)

    # Get the cluster labels and centroids
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Create empty lists to store the clusters and their min/max values
    clusters = [[] for _ in range(num_clusters)]
    min_values = [float('inf')] * num_clusters
    max_values = [float('-inf')] * num_clusters

    # Iterate through the data and assign each point to its corresponding cluster
    for i, label in enumerate(labels):
        clusters[label].append(unique_arr[i])
        if unique_arr[i] < min_values[label]:
            min_values[label] = unique_arr[i]
        if unique_arr[i] > max_values[label]:
            max_values[label] = unique_arr[i]

    return clusters, min_values, max_values, num_clusters


# Print the clusters and their corresponding minimum and maximum values
#for i in range(len(clusters)):
    #print(f"Cluster {i+1}: {clusters[i]}")
   # print(f"Min: {min_vals[i]}, Max: {max_vals[i]}")

def cluster_check(val,min,max,n):  # function to check a value lies in which cluster
    for i in range(n):
        if(min[i]<=val and val<=max[i]):
            return i


#min_vals and max_vals contain the minimum and maximum values values in each cluster
clusters, min_vals99, max_vals99,no_of_clusters99  = create_clusters(sorted_array)
min_vals99 = np.sort(min_vals99)
max_vals99 = np.sort(max_vals99)


for i in range(frame_height): #iterating along the rows
    for j in range(frame_width): #iterating along the columns of the ith row
        if ((0.01 * maximum_value_in_data) < blank[i][j]): #considering maximum 99% values
            cluster = cluster_check(blank[i][j], min_vals99, max_vals99,no_of_clusters99)
            color = interpolate(cluster)
            overlay[i, j] = color # the pixel value of i,j postion is changed to the BGR values of color

## clustering values less than 1 percent
clusters, min_vals01, max_vals01,no_of_clusters01 = create_clusters(small_values)
min_vals01 = np.sort(min_vals01)
max_vals01 = np.sort(max_vals01)

for i in range(frame_height): #iterating along the rows
    for j in range(frame_width): #iterating along the columns of the ith row
        if((0<blank[i][j]) and (blank[i][j]<0.01*maximum_value_in_data)): #considering minimum 1% values
            cluster = cluster_check(blank[i][j], min_vals01, max_vals01, no_of_clusters01)
            color = interpolate2(cluster)
            overlay[i, j] = color
cv2.imshow("overlay", overlay)
#Following line is used to make heatmaps transparent
output_image = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

for i in range(len(section_count)):
    print("Time spent in aisle ",i+1,"is ",section_count[i]/30)




cv2.imshow("output image", output_image)
cv2.imshow("frame",frame)
cv2.waitKey(0)
cv2.destroyAllWindows()