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

        self.image = None
        self.landmarks = None
        self.lmrk_d = dict()
        for id,lm in enumerate((self.mp_pose).PoseLandmark):
            self.lmrk_d[str(lm)[13::]] = id

    def draw_all_landmarks(self, results):
        """
        Draws all the self.landmarks on the self.image. The
        method uses mediapipe's built-in utilities
        """
        self.mp_drawing.draw_landmarks(
            self.image,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing_styles.get_default_pose_landmarks_style())

    def show_fps(self, prev_time):
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(self.image, (str(int(fps))), (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA, False)
        
        return prev_time

    # TODO find a way to update the plot with the new self.landmarks instaed of creating a new one
    def create_plot(self, vis_threshold=0.9):
        fig = plt.figure(figsize=(5,5))
        ax = plt.axes(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        if self.landmarks:
            for key in self.lmrk_d.keys():
                point = self.landmarks[self.lmrk_d[key]]
                if point.visibility > vis_threshold:
                    ax.scatter(point.x,point.y,point.z)
                    
        plt.show()
        fig.canvas.draw()

    def process_landmarks(self, results, color=(0,0,255), thickness=2, draw = True):
        """
        Sets self.lamndmarks if the results are valid
        Draws and links shoulders, hips, and neck self.landmarks 
        and shows angle between shoulders and head
        """
        try:
            self.landmarks = results.pose_landmarks.landmark
        except:
            self.landmarks = None
        
        if not draw:
            return
    
        if self.landmarks:
            h,w,d =(self.image).shape
            L_S = self.lmrk_d['LEFT_SHOULDER'] #11
            R_S = self.lmrk_d['RIGHT_SHOULDER'] #12
            L_H = self.lmrk_d['LEFT_HIP'] #23
            R_H = self.lmrk_d['RIGHT_HIP'] #24
            L_E = self.lmrk_d['LEFT_EAR'] #7
            R_E = self.lmrk_d['RIGHT_EAR'] #8
            
            # draw points on shoulders, hips, ears
            lst = [L_S,R_S,L_H,R_H,L_E,R_E]
            for id in lst:
                lm = self.landmarks[id]
                cx, cy = int(lm.x*w), int(lm.y*h)
                cv2.circle(img=self.image, center=(cx,cy), radius=3, color=color, thickness=thickness)

            # draw shoulder to shoulder
            cv2.line(self.image, 
                (int(self.landmarks[L_S].x*w), int(self.landmarks[L_S].y*h)), 
                (int(self.landmarks[R_S].x*w), int(self.landmarks[R_S].y*h)), 
                color, thickness)
            # draw left shoulder to left hip
            cv2.line(self.image, 
                (int(self.landmarks[L_S].x*w), int(self.landmarks[L_S].y*h)), 
                (int(self.landmarks[L_H].x*w), int(self.landmarks[L_H].y*h)), 
                color, thickness)
            # draw right shoulder to right hip
            cv2.line(self.image, 
                (int(self.landmarks[R_H].x*w), int(self.landmarks[R_H].y*h)), 
                (int(self.landmarks[R_S].x*w), int(self.landmarks[R_S].y*h)), 
                color, thickness)
            # draw left hip to right hip
            cv2.line(self.image, 
                (int(self.landmarks[L_H].x*w), int(self.landmarks[L_H].y*h)), 
                (int(self.landmarks[R_H].x*w), int(self.landmarks[R_H].y*h)), 
                color, thickness)
            # draw left ear to right ear
            cv2.line(self.image, 
                (int(self.landmarks[L_E].x*w), int(self.landmarks[L_E].y*h)), 
                (int(self.landmarks[R_E].x*w), int(self.landmarks[R_E].y*h)), 
                color, thickness)
            
            # if both ears visible, draw neck line between them
            if self.landmarks[L_E].visibility > 0.95 and self.landmarks[R_E].visibility > 0.95:
                cv2.line(self.image, 
                    (int((self.landmarks[R_S].x+self.landmarks[L_S].x)*w/2), int((self.landmarks[R_S].y+self.landmarks[L_S].y)*h/2)), 
                    (int((self.landmarks[R_E].x+self.landmarks[L_E].x)*w/2), int((self.landmarks[R_E].y+self.landmarks[L_E].y)*h/2)), 
                    color, thickness)
            # if only left ear visible, draw neck line to left ear
            elif self.landmarks[L_E].visibility > 0.95:
                cv2.line(self.image, 
                    (int((self.landmarks[R_S].x+self.landmarks[L_S].x)*w/2), int((self.landmarks[R_S].y+self.landmarks[L_S].y)*h/2)), 
                    (int((self.landmarks[L_E].x)*w), int((self.landmarks[L_E].y)*h)), 
                    color, thickness)
            # if only right ear visible, draw neck line to right ear
            elif self.landmarks[R_E].visibility > 0.95:
                cv2.line(self.image, 
                    (int((self.landmarks[R_S].x+self.landmarks[L_S].x)*w/2), int((self.landmarks[R_S].y+self.landmarks[L_S].y)*h/2)), 
                    (int((self.landmarks[R_E].x)*w), int((self.landmarks[R_E].y)*h)), 
                    color, thickness)
            else:
                print("NO EARS DETECTED")

            

    # TODO change to find angle between a point in front of person's chest in the center,
    # point middle point between two shoulders, and middle point between two ears

    def neck_posture_angle(self, color=(0,0,255)):
        if not self.landmarks:
            return
        L_S = self.lmrk_d['LEFT_SHOULDER'] #11
        R_S = self.lmrk_d['RIGHT_SHOULDER'] #12
        L_E = self.lmrk_d['LEFT_EAR'] #7
        R_E = self.lmrk_d['RIGHT_EAR'] #8

        a = [self.landmarks[R_S].x,self.landmarks[R_S].y,self.landmarks[R_S].z]
        b = [(self.landmarks[R_S].x+self.landmarks[L_S].x)/2,(self.landmarks[R_S].y+self.landmarks[L_S].y)/2,(self.landmarks[R_S].z+self.landmarks[L_S].z)/2]
        c = [(self.landmarks[R_E].x+self.landmarks[L_E].x)/2,(self.landmarks[R_E].y+self.landmarks[L_E].y)/2,(self.landmarks[R_E].z+self.landmarks[L_E].z)/2]
        h,w,_ = (self.image).shape
        three_d_angle = self.get_angle_3d(a,b,c)
        two_d_angle = self.get_angle_2D(a,b,c)
        if two_d_angle > 90:
            two_d_angle = 180 - two_d_angle
        color=(0,255,0)
        cv2.putText(self.image, "3D angle: "+str(three_d_angle), (int((self.landmarks[L_S].x+self.landmarks[R_S].x)*w/2), int(self.landmarks[L_S].y*h)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(self.image, "2D angle: "+str(two_d_angle), (int((self.landmarks[L_S].x+self.landmarks[R_S].x)*w/2), int(self.landmarks[L_S].y*h+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        
    def get_angle_3d(self, a, b, c):
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

    def get_angle_2D(self, a, b, c):
        # only take the x and y coordinates
        a = np.array(a[:2])
        b = np.array(b[:2])
        c = np.array(c[:2])

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)

        return round(np.degrees(angle),2)

def main():
    detector = detectPose()

    # cap = cv2.VideoCapture("video_samples/6.mp4")
    cap = cv2.VideoCapture(0)

    prev_time = time.time()

    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("End video. Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        detector.image = image
        (detector.image).flags.writeable = False
        detector.image = cv2.cvtColor(detector.image, cv2.COLOR_BGR2RGB)
        results = detector.pose.process(detector.image)

        # Draw the pose annotation on the self.image.
        (detector.image).flags.writeable = True
        detector.image = cv2.cvtColor((detector.image), cv2.COLOR_RGB2BGR)

        prev_time = detector.show_fps(prev_time) # Shows FPS before reasssigning prev_time
        
        detector.process_landmarks(results)
        detector.neck_posture_angle()
        # cv2.imshow('MediaPipe Pose', cv2.flip(self.image, 1))
        cv2.imshow('MediaPipe Pose', detector.image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()