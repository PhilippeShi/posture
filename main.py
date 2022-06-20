import cv2
import time
from pose_module import detectPose

def main():
    detector = detectPose(show_video_image=False)

    # cap = cv2.VideoCapture("video_samples/6.mp4")
    cap = cv2.VideoCapture(0)

    prev_time = time.time()

    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("End video. Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        detector.set_images(image)

        (detector.image).flags.writeable = False
        detector.image = cv2.cvtColor(detector.image, cv2.COLOR_BGR2RGB)
        results = detector.pose.process(detector.image)

        # Draw the pose annotation on the self.image.
        (detector.image).flags.writeable = True
        detector.image = cv2.cvtColor((detector.image), cv2.COLOR_RGB2BGR)

        prev_time = detector.show_fps(prev_time) # Shows FPS before reasssigning prev_time
        
        detector.process_landmarks(results)
        detector.neck_posture_angle()        
        detector.show()
        # cv2.imshow("Image 1", detector.image)
        # cv2.imshow("Image 2", detector.blank_image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()