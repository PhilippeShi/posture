from cv2 import cv2
import numpy as np
import os
import mediapipe as mp

# Load Yolo
dir = os.path.dirname(os.path.realpath(__file__))
# weights = os.path.join(dir, "YOLO/yolov3.weights")
# cfg = os.path.join(dir, "YOLO/yolov3.cfg")
# weights = os.path.join(dir, "YOLO/yolov3-tiny.weights")
# cfg = os.path.join(dir, "YOLO/yolov3-tiny.cfg")
weights = os.path.join(dir, "YOLO/yolov2-tiny-voc.weights")
cfg = os.path.join(dir, "YOLO/yolov2-tiny-voc.cfg")
names = os.path.join(dir, "YOLO/coco.names")
net = cv2.dnn.readNet(weights, cfg)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose =  mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

classes = []
with open(names, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
vid = os.path.join(dir, "1.mp4")
vid = ("C:/Users/phili/Documents/GitHub/posturebiofeedback/video_samples/4.mp4")
cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("End video. Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue
    
    # image = cv2.resize(image, None, fx=0.4, fy=0.4)
    height, width, channels = image.shape

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

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
        # class_id[i] == 14 is person for yolov2-tiny
        if i in indexes and class_ids[i] == 14:
            x, y, w, h = boxes[i]
            # label = str(classes[class_ids[i]])
            # color = colors[i]
            # cv2.putText(image, label, (x, y + 30), font, 3, color, 3)
            color = (255, 255, 255)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cropped = image[y:y + h, x:x + w]
            crops.append((cropped, (y,x)))
            
    for crop, (y,x) in crops:
        try:    
            # To improve performance, optionally mark the crop as not writeable to
            # pass by reference.
            crop.flags.writeable = False
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            results = pose.process(crop)

            # Draw the pose annotation on the crop.
            crop.flags.writeable = True
            crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                crop,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # Flip the crop horizontally for a selfie-view display.
            # cv2.imshow('MediaPipe Pose', cv2.flip(crop, 1))
            # overlaying cropped image on original image
            image[y:y+crop.shape[0], x:x+crop.shape[1]] = crop
            
        except:
            pass
    cv2.imshow("Pose", image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

