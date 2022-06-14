import cv2
import time
from pose_module import detectPose

def main():
    detector = detectPose()

    cap = cv2.VideoCapture("video_samples/6.mp4")
    # cap = cv2.VideoCapture(0)

    prev_time = time.time()

    with detector.pose as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("End video. Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break
        
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            
            prev_time = detector.show_fps(image, prev_time)
            detector.draw_landmarks(image, results)

            # cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            cv2.imshow('MediaPipe Pose', image)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()