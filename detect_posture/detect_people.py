import cv2
import os
import numpy as np
import mediapipe as mp

class detect_people:
    def __init__(self):
        # Load Yolo
        dir = os.path.dirname(os.path.realpath(__file__))
        cfg = os.path.join(dir, "YOLO/yolov2-tiny-voc.cfg")
        weights = os.path.join(dir, "YOLO/yolov2-tiny-voc.weights")
        names = os.path.join(dir, "YOLO/coco.names")

        self.net = cv2.dnn.readNet(weights, cfg)
        classes = []
        with open(names, "r") as f:
            classes = [line.strip() for line in f.readlines()]
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(classes), 3))

    def box_person(self, image, landmarks):

        height, width, channels = image.shape
        # Detecting objects
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

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
                # TODO check if detected landmarks are in the box
                # if so, then ignore that detected person
                
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
                image[y:y+crop.shape[0], x:x+crop.shape[1]] = crop
                
            except:
                pass