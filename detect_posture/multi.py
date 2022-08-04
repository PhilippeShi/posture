import cv2
from tf_bodypix.api import load_model, download_model, BodyPixModelPaths
import numpy as np
import os

bp_model = load_model(download_model(
    BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16))

dir = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(dir, "1.mp4")
cap = cv2.VideoCapture(path)
while cap.isOpened():

    success, image = cap.read()
    if not success:
        print("error")
        break

    prediction = bp_model.predict_single(image)
    mask = prediction.get_mask(threshold=0.01).numpy().astype(np.uint8)
    new_mask = cv2.bitwise_and(image, image, mask=mask)

    cv2.imshow("dsa", new_mask)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
