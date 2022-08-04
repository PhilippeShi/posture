from cv2 import cv2
import numpy as np
import os

# Load Yolo
dir = os.path.dirname(os.path.realpath(__file__))
# weights = os.path.join(dir, "yolov3.weights")
# cfg = os.path.join(dir, "yolov3.cfg")
# weights = os.path.join(dir, "yolov3-tiny.weights")
# cfg = os.path.join(dir, "yolov3-tiny.cfg")
weights = os.path.join(dir, "yolov2-tiny-voc.weights")
cfg = os.path.join(dir, "yolov2-tiny-voc.cfg")
names = os.path.join(dir, "coco.names")
net = cv2.dnn.readNet(weights, cfg)
classes = []
with open(names, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
vid = os.path.join(dir, "1.mp4")
cap = cv2.VideoCapture(1)
count = 0
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("End video. Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue
    
    
    # image = cv2.resize(image, None, fx=0.4, fy=0.4)
    height, width, channels = image.shape


    # Detecting objects
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    crops = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            # color = colors[i]
            color = (255, 255, 255)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            crops.append(image[y:y+h, x:x+w])
            cv2.putText(image, label, (x, y + 30), font, 3, color, 3)
    cv2.imshow("Image", image)
    for i, crop in enumerate(crops):
        try:
            cv2.imshow(f"Crop {i}", crop)
        except:
            pass
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    count += 1

cap.release()
cv2.destroyAllWindows()

