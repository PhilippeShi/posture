import time
import cv2
import mediapipe as mp
import numpy as np
from .utils import get_angle, get_percent_difference, track_person, overlay_image

class detectPose():
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5, show_video_image=True):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
    
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)

        self.image = None
        self.blank_image = None
        self.show_video_image = show_video_image

        self.landmarks = None
        self.lmrk_d = dict()
        for id,lm in enumerate((self.mp_pose).PoseLandmark):
            self.lmrk_d[str(lm)[13::]] = id
        
        self.bad_postures = list()

    def set_images(self, image):
        self.image = image
        self.blank_image = np.zeros(self.image.shape, dtype=np.uint8)


    def images(self):
        if self.show_video_image:
            return [self.image, self.blank_image]
        else:
            return [self.blank_image]

    def draw_all_landmarks(self, results):
        """
        Method uses mediapipe's built-in utilities. Sometimes creates inconsistent 
        coloring of the video images.
        """
        for img in self.images():
            self.mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing_styles.get_default_pose_landmarks_style())

    def show_fps(self, prev_time):
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        for img in self.images():
            cv2.putText(img, (str(int(fps))), (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA, False)
        
        return prev_time

    def show(self):
        """ Shows video results for quick visualization and prototyping.
        """
        for count, img in enumerate(self.images()):
            cv2.imshow("Image "+str(count+1), img)
 
    def change_subject(self):
        if not self.landmarks:
            return
        person, coords = track_person(self.image, self.landmarks)
        person_blurred = cv2.blur(person, (1000,1000))
        self.image = overlay_image(self.image.copy(), person_blurred, coords[0], coords[1])
        self.prev_blurred_coords = coords
    
    def blur_all_except_person(self):
        if self.landmarks is None:
            return
        
        person, coords = track_person(self.image, self.landmarks)
        self.image = cv2.blur(self.image.copy(), (1000,1000))
        self.image = overlay_image(self.image.copy(), person, coords[0], coords[1])
        cv2.imshow("Image", self.image)

    def blur_person(self):
        if not self.landmarks:
            return
        
        person, coords = track_person(self.image, self.landmarks)
        blurred_person = cv2.blur(person.copy(), (1000,1000))
        try:
            self.image = overlay_image(self.image.copy(), blurred_person, coords[0], coords[1])
            cv2.imshow("Image", self.image)
        except:
            pass
        
    def process_landmarks(self, results, color=(0,0,255), thickness=2, draw=True, vis_threshold=0.5):
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
            h,w,_ =(self.image).shape
            L_S = self.lmrk_d['LEFT_SHOULDER'] #11
            R_S = self.lmrk_d['RIGHT_SHOULDER'] #12
            L_H = self.lmrk_d['LEFT_HIP'] #23
            R_H = self.lmrk_d['RIGHT_HIP'] #24
            L_E = self.lmrk_d['LEFT_EAR'] #7
            R_E = self.lmrk_d['RIGHT_EAR'] #8
            L_K = self.lmrk_d['LEFT_KNEE'] #25
            R_K = self.lmrk_d['RIGHT_KNEE'] #26
            L_A = self.lmrk_d['LEFT_ANKLE'] #27
            R_A = self.lmrk_d['RIGHT_ANKLE'] #28

            # draw points on shoulders, hips, ears
            for image in self.images():
                lst = [L_S, R_S, L_H, R_H, L_E, R_E, L_K, R_K, L_A, R_A]
                for id in lst:
                    lm = self.landmarks[id]
                    if lm.visibility > vis_threshold:
                        cx, cy = int(lm.x*w), int(lm.y*h)
                        cv2.circle(image, center=(cx,cy), radius=3, color=color, thickness=thickness)
                
                # draw shoulder to shoulder
                cv2.line(image, 
                    (int(self.landmarks[L_S].x*w), int(self.landmarks[L_S].y*h)), 
                    (int(self.landmarks[R_S].x*w), int(self.landmarks[R_S].y*h)), 
                    color, thickness)
                # draw left shoulder to left hip
                cv2.line(image, 
                    (int(self.landmarks[L_S].x*w), int(self.landmarks[L_S].y*h)), 
                    (int(self.landmarks[L_H].x*w), int(self.landmarks[L_H].y*h)), 
                    color, thickness)
                # draw right shoulder to right hip
                cv2.line(image, 
                    (int(self.landmarks[R_H].x*w), int(self.landmarks[R_H].y*h)), 
                    (int(self.landmarks[R_S].x*w), int(self.landmarks[R_S].y*h)), 
                    color, thickness)
                # draw left hip to right hip
                cv2.line(image, 
                    (int(self.landmarks[L_H].x*w), int(self.landmarks[L_H].y*h)), 
                    (int(self.landmarks[R_H].x*w), int(self.landmarks[R_H].y*h)), 
                    color, thickness)
                # draw left hip to left knee
                if self.landmarks[L_K].visibility > vis_threshold:
                    cv2.line(image, 
                        (int(self.landmarks[L_H].x*w), int(self.landmarks[L_H].y*h)), 
                        (int(self.landmarks[L_K].x*w), int(self.landmarks[L_K].y*h)), 
                        color, thickness)
                # draw right hip to right knee
                if self.landmarks[R_K].visibility > vis_threshold:
                    cv2.line(image, 
                        (int(self.landmarks[R_H].x*w), int(self.landmarks[R_H].y*h)), 
                        (int(self.landmarks[R_K].x*w), int(self.landmarks[R_K].y*h)), 
                        color, thickness)
                # draw left knee to left ankle
                if self.landmarks[L_A].visibility > vis_threshold:
                    cv2.line(image, 
                        (int(self.landmarks[L_K].x*w), int(self.landmarks[L_K].y*h)), 
                        (int(self.landmarks[L_A].x*w), int(self.landmarks[L_A].y*h)), 
                        color, thickness)
                # draw right knee to right ankle
                if self.landmarks[R_A].visibility > vis_threshold:
                    cv2.line(image, 
                        (int(self.landmarks[R_K].x*w), int(self.landmarks[R_K].y*h)), 
                        (int(self.landmarks[R_A].x*w), int(self.landmarks[R_A].y*h)), 
                        color, thickness)
                # draw left ear to right ear
                cv2.line(image, 
                    (int(self.landmarks[L_E].x*w), int(self.landmarks[L_E].y*h)), 
                    (int(self.landmarks[R_E].x*w), int(self.landmarks[R_E].y*h)), 
                    color, thickness)
        
                # if both ears visible, draw neck line between them
                if self.landmarks[L_E].visibility > vis_threshold and self.landmarks[R_E].visibility > vis_threshold:
                    cv2.line(image, 
                        (int((self.landmarks[R_S].x+self.landmarks[L_S].x)*w/2), 
                        int((self.landmarks[R_S].y+self.landmarks[L_S].y)*h/2)), 
                        (int((self.landmarks[R_E].x+self.landmarks[L_E].x)*w/2), 
                        int((self.landmarks[R_E].y+self.landmarks[L_E].y)*h/2)), 
                        color, thickness)
                # if only left ear visible, draw neck line to left ear
                elif self.landmarks[L_E].visibility > vis_threshold:
                    cv2.line(image, 
                        (int((self.landmarks[R_S].x+self.landmarks[L_S].x)*w/2), 
                        int((self.landmarks[R_S].y+self.landmarks[L_S].y)*h/2)), 
                        (int((self.landmarks[L_E].x)*w), int((self.landmarks[L_E].y)*h)), 
                        color, thickness)
                # if only right ear visible, draw neck line to right ear
                elif self.landmarks[R_E].visibility > vis_threshold:
                    cv2.line(image, 
                        (int((self.landmarks[R_S].x+self.landmarks[L_S].x)*w/2), 
                        int((self.landmarks[R_S].y+self.landmarks[L_S].y)*h/2)), 
                        (int((self.landmarks[R_E].x)*w), int((self.landmarks[R_E].y)*h)), 
                        color, thickness)
                else:
                    print("NO EARS DETECTED")

    def detect_orientation(self, shoulder_height_variation_threshold):
        """
        Detects a subject's orientation to the camera. Uses the percentage 
        difference between the shoulders' heights (y-coordinates).

        CURRENTLY NOT IN USE, see detect_orientation_2
        """
        if not self.landmarks:
            return
        
        # Using shoulders as reference
        L_S = self.lmrk_d['LEFT_SHOULDER'] #11
        R_S = self.lmrk_d['RIGHT_SHOULDER'] #12

        L_Y = self.landmarks[L_S].y
        R_Y = self.landmarks[R_S].y

        # Calculate the % variation between the shoulders' y-coordinates
        diff = abs((L_Y-R_Y)/((L_Y+R_Y)*2))

        if diff < shoulder_height_variation_threshold:
            return "front", round(diff*100,2)
        else:
            return "side", round(diff*100,2)

    def detect_orientation_2(self, shoulder_hip_ratio_threshold):
        if not self.landmarks:
            return
        
        ratio = self.shoulders_hips_ratio()

        if ratio > shoulder_hip_ratio_threshold:
            # for img in self.images():
            #     cv2.putText(img, "front"+str(round(ratio,2)), (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            return "front", round(ratio,3)
        else:
            # for img in self.images():
            #     cv2.putText(img, "side"+str(round(ratio,2)), (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            return "side", round(ratio,3)
    
    def shoulders_hips_ratio(self):
        if not self.landmarks:
            return
        
        # Using shoulders as reference
        L_S = self.lmrk_d['LEFT_SHOULDER']
        R_S = self.lmrk_d['RIGHT_SHOULDER']
        L_H = self.lmrk_d['LEFT_HIP']
        R_H = self.lmrk_d['RIGHT_HIP']

        LS = [self.landmarks[L_S].x, self.landmarks[L_S].y]
        RS = [self.landmarks[R_S].x, self.landmarks[R_S].y]
        length_shoulders = np.linalg.norm(np.array(LS)-np.array(RS))

        LH = [self.landmarks[L_H].x, self.landmarks[L_H].y]
        RH = [self.landmarks[R_H].x, self.landmarks[R_H].y]

        left_side = np.linalg.norm(np.array(LH)-np.array(LS))
        right_side = np.linalg.norm(np.array(RH)-np.array(RS))
        hips_shoulders_length = (left_side + right_side)/2

        return length_shoulders/hips_shoulders_length

    def neck_posture(self, color=(0,0,255), auto_detect_orientation=False, 
        shoulder_height_variation_threshold=0.018, 
        shoulder_hip_ratio_threshold=0.45,
        neck_ratio_threshold=0.65, 
        neck_angle_threshold=40, put_orientation_text=False):
        """ 
        auto_detect_orientation has to be True to use neck_ratio_threshold and neck_angle_threshold
        """
        if not self.landmarks:
            return
        L_S = self.lmrk_d['LEFT_SHOULDER'] #11
        R_S = self.lmrk_d['RIGHT_SHOULDER'] #12
        L_E = self.lmrk_d['LEFT_EAR'] #7
        R_E = self.lmrk_d['RIGHT_EAR'] #8

        a = [self.landmarks[R_S].x, self.landmarks[R_S].y, self.landmarks[R_S].z]
        a_left = [self.landmarks[L_S].x, self.landmarks[L_S].y, self.landmarks[L_S].z]
        
        # point between shoulders
        b = [(self.landmarks[R_S].x+self.landmarks[L_S].x)/2,
            (self.landmarks[R_S].y+self.landmarks[L_S].y)/2,
            (self.landmarks[R_S].z+self.landmarks[L_S].z)/2]

        # point between ears
        c = [(self.landmarks[R_E].x+self.landmarks[L_E].x)/2,
            (self.landmarks[R_E].y+self.landmarks[L_E].y)/2,
            (self.landmarks[R_E].z+self.landmarks[L_E].z)/2]

        # TODO miror image to get the correct orientation (left-right)
        h,w,_ = (self.image).shape
        three_d_angle = get_angle(a,b,c,3)

        left_length = np.linalg.norm(np.array(a)-np.array(c))
        right_length = np.linalg.norm(np.array(a_left)-np.array(c))

        if left_length < right_length:
            two_d_angle = get_angle(a_left,b,c,2)
        else:
            two_d_angle = get_angle(a,b,c,2)


        ratio = self.neck_shoulders_ratio()

        if two_d_angle > 90:
            two_d_angle = 180 - two_d_angle
        if three_d_angle > 90:
            three_d_angle = 180 - three_d_angle

        color=(0,255,0)
        if shoulder_hip_ratio_threshold is not None:
            detected_orientation, diff = self.detect_orientation_2(shoulder_hip_ratio_threshold)
        else:
            detected_orientation, diff = self.detect_orientation(shoulder_height_variation_threshold)

        good_posture = True

        if put_orientation_text:
            for img in self.images():
                cv2.putText(img, f"{detected_orientation} {str(diff)}", (0, 70), 
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        if auto_detect_orientation:
            if detected_orientation == "front":
                if ratio < neck_ratio_threshold:
                    color = (255,0,0)
                    good_posture = False
                for img in self.images():
                    cv2.putText(img, "ratio: "+str(round(ratio,2)), 
                        (int((self.landmarks[L_S].x+self.landmarks[R_S].x)*w/2), int(self.landmarks[L_S].y*h+20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
            elif detected_orientation == "side":
                if two_d_angle < neck_angle_threshold:
                    good_posture = False
                    color = (255,0,0)
                for img in self.images():
                    cv2.putText(img, "angle: "+str(round(two_d_angle,2)), 
                        (int((self.landmarks[L_S].x+self.landmarks[R_S].x)*w/2), int(self.landmarks[L_S].y*h+20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            

        else:
            two_d_color, ratio_color = (0,255,0), (0,255,0)
            if two_d_angle < neck_angle_threshold:
                two_d_color = (255,0,0)
            if ratio < neck_ratio_threshold:
                ratio_color = (255,0,0)
            for img in self.images():
                cv2.putText(img, "ratio: "+str(round(ratio,2)), 
                    (int((self.landmarks[L_S].x+self.landmarks[R_S].x)*w/2), int(self.landmarks[L_S].y*h+20)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, ratio_color, 2)
                cv2.putText(img, "2D angle: "+str(round(two_d_angle,2)), 
                    (int((self.landmarks[L_S].x+self.landmarks[R_S].x)*w/2), int(self.landmarks[L_S].y*h)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, two_d_color, 2)
                cv2.putText(img, "3D angle: "+str(round(three_d_angle,2)), 
                    (int((self.landmarks[L_S].x+self.landmarks[R_S].x)*w/2), int(self.landmarks[L_S].y*h-20)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
        return good_posture, detected_orientation, ratio, two_d_angle, three_d_angle

    def neck_shoulders_ratio(self):
        L_S = self.lmrk_d['LEFT_SHOULDER'] #11
        R_S = self.lmrk_d['RIGHT_SHOULDER'] #12
        L_E = self.lmrk_d['LEFT_EAR'] #7
        R_E = self.lmrk_d['RIGHT_EAR'] #8
        
        a = [self.landmarks[L_S].x,self.landmarks[L_S].y]
        b = [self.landmarks[R_S].x,self.landmarks[R_S].y]
        length_shoulders = np.linalg.norm(np.array(a)-np.array(b))

        c = [(self.landmarks[R_S].x+self.landmarks[L_S].x)/2, (self.landmarks[R_S].y+self.landmarks[L_S].y)/2]
        d = [(self.landmarks[R_E].x+self.landmarks[L_E].x)/2, (self.landmarks[R_E].y+self.landmarks[L_E].y)/2]
        
        length_neck = np.linalg.norm(np.array(c)-np.array(d))
        return length_neck/length_shoulders

    def set_bad_posture(self, neck_ratio, neck_angle):
        if not self.landmarks:
            return
        
        if not self.check_bad_posture(neck_ratio, neck_angle):
            self.bad_postures.append((neck_ratio, neck_angle))
            print("added bad posture")
        else:
            print("bad posture already exists")
        
    def check_bad_posture(self, neck_ratio, neck_angle, variation=0.1):
        if not self.landmarks:
            return
        
        for (r,a) in self.bad_postures:
            if get_percent_difference(r, neck_ratio) < variation*100 and get_percent_difference(a, neck_angle) < variation*100:
                return True
        return False

def main():
    detector = detectPose(show_video_image=True)

    # cap = cv2.VideoCapture("video_samples/6.mp4")
    cap = cv2.VideoCapture(0)

    prev_time = time.time()
    time_bad_posture = 0
    good_posture = True

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("End video. Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        detector.set_images(image)
        results = detector.pose.process(detector.image)
        
        detector.process_landmarks(results, vis_threshold=0.7)
        # detector.draw_all_landmarks(results)
        good_posture = detector.neck_posture(
            auto_detect_orientation=True, 
            neck_ratio_threshold=0.70, 
            neck_angle_threshold=60,
            put_orientation_text=True
            ) 
        
        detector.detect_orientation_2(shoulder_hip_ratio_threshold=0.45)

        if good_posture:
            time_bad_posture = 0

        elif not good_posture:
            time_bad_posture += time.time() - prev_time
            cv2.putText(detector.image, str(int(time_bad_posture)), (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            if time_bad_posture > 1:
                cv2.putText(detector.image, f"MORE THAN {int(time_bad_posture)}s!", (0,150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 10)


        prev_time = detector.show_fps(prev_time) # Shows FPS before reasssigning prev_time
        detector.show()
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

    # TODO - Add alert for when bad posture is detected for more than X seconds