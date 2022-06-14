import time
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np

class detectPose():
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
    
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)
        
        self.lmrk_d = dict()
        for id,lm in enumerate((self.mp_pose).PoseLandmark):
            self.lmrk_d[str(lm)[13::]] = id

    def draw_all_landmarks(self, image, results):
        """
        Draws all the landmarks on the image. The
        method uses mediapipe's built-in utilities
        """
        self.mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing_styles.get_default_pose_landmarks_style())

    def show_fps(self, image, prev_time):
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(image, (str(int(fps))), (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA, False)
        
        return prev_time

    # TODO find a way to update the plot with the new landmarks instaed of creating a new one
    def create_plot(self, landmarks, vis_threshold=0.9):
        fig = plt.figure(figsize=(5,5))
        ax = plt.axes(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        if landmarks:
            for key in self.lmrk_d.keys():
                point = landmarks[self.lmrk_d[key]]
                if point.visibility > vis_threshold:
                    ax.scatter(point.x,point.y,point.z)
                    
        plt.show()
        fig.canvas.draw()

    def draw_landmarks(self, image, results, color=(0,0,255), thickness=2):
        """
        Draws and links shoulders, hips, and neck landmarks 
        and shows angle between shoulders and head
        """
        try:
            landmarks = results.pose_landmarks.landmark
        except:
            landmarks = None
    
        if landmarks:
            h,w,d = image.shape
            
            L_S = self.lmrk_d['LEFT_SHOULDER'] #11
            R_S = self.lmrk_d['RIGHT_SHOULDER'] #12
            L_H = self.lmrk_d['LEFT_HIP'] #23
            R_H = self.lmrk_d['RIGHT_HIP'] #24
            L_E = self.lmrk_d['LEFT_EAR'] #7
            R_E = self.lmrk_d['RIGHT_EAR'] #8

            lst = [L_S,R_S,L_H,R_H,L_E,R_E]
            for id in lst:
                lm = landmarks[id]
                cx, cy = int(lm.x*w), int(lm.y*h)
                cv2.circle(img=image, center=(cx,cy), radius=3, color=color, thickness=thickness)

            cv2.line(image, 
                (int(landmarks[L_S].x*w), int(landmarks[L_S].y*h)), 
                (int(landmarks[R_S].x*w), int(landmarks[R_S].y*h)), 
                color, thickness)
            cv2.line(image, 
                (int(landmarks[L_S].x*w), int(landmarks[L_S].y*h)), 
                (int(landmarks[L_H].x*w), int(landmarks[L_H].y*h)), 
                color, thickness)
            cv2.line(image, 
                (int(landmarks[R_H].x*w), int(landmarks[R_H].y*h)), 
                (int(landmarks[R_S].x*w), int(landmarks[R_S].y*h)), 
                color, thickness)
            cv2.line(image, 
                (int(landmarks[L_H].x*w), int(landmarks[L_H].y*h)), 
                (int(landmarks[R_H].x*w), int(landmarks[R_H].y*h)), 
                color, thickness)
            cv2.line(image, 
                (int(landmarks[L_E].x*w), int(landmarks[L_E].y*h)), 
                (int(landmarks[R_E].x*w), int(landmarks[R_E].y*h)), 
                color, thickness)
            
            if landmarks[L_E].visibility > 0.95 and landmarks[R_E].visibility > 0.95:
                cv2.line(image, 
                    (int((landmarks[R_S].x+landmarks[L_S].x)*w/2), int((landmarks[R_S].y+landmarks[L_S].y)*h/2)), 
                    (int((landmarks[R_E].x+landmarks[L_E].x)*w/2), int((landmarks[R_E].y+landmarks[L_E].y)*h/2)), 
                    color, thickness)

            elif landmarks[L_E].visibility > 0.95:
                cv2.line(image, 
                    (int((landmarks[R_S].x+landmarks[L_S].x)*w/2), int((landmarks[R_S].y+landmarks[L_S].y)*h/2)), 
                    (int((landmarks[L_E].x)*w), int((landmarks[L_E].y)*h)), 
                    color, thickness)

            elif landmarks[R_E].visibility > 0.95:
                cv2.line(image, 
                    (int((landmarks[R_S].x+landmarks[L_S].x)*w/2), int((landmarks[R_S].y+landmarks[L_S].y)*h/2)), 
                    (int((landmarks[R_E].x)*w), int((landmarks[R_E].y)*h)), 
                    color, thickness)
            else:
                print("NO EARS DETECTED")
            
            a = [landmarks[R_S].x,landmarks[R_S].y,landmarks[R_S].z]
            b = [(landmarks[R_S].x+landmarks[L_S].x)/2,(landmarks[R_S].y+landmarks[L_S].y)/2,(landmarks[R_S].z+landmarks[L_S].z)/2]
            c = [(landmarks[R_E].x+landmarks[L_E].x)/2,(landmarks[R_E].y+landmarks[L_E].y)/2,(landmarks[R_E].z+landmarks[L_E].z)/2]

            cv2.putText(image, str(get_angle_3d(a,b,c)), (int((landmarks[L_S].x+landmarks[R_S].x)*w/2), int(landmarks[L_S].y*h)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)


def get_angle_3d(a, b, c):
    """
    Helper method that takes 3 lists of 3D coordinates
    and return the angle in the 3D space.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    rad_angle = np.arccos(cosine_angle)
    deg_angle = np.degrees(rad_angle)

    if deg_angle > 180:
        deg_angle = 360 - deg_angle
    return round(deg_angle,2)

def main():
    detector = detectPose()

    # cap = cv2.VideoCapture("6.mp4")
    cap = cv2.VideoCapture(0)

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