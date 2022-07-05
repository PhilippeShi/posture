import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk
import time
from pose_module import detectPose
import os

class App:
    def __init__(self, window, window_title, 
            video_source=0, 
            show_video=False, 
            auto_detect_orientation=False,
            draw_all_landmarks=False,
            draw_pose_landmarks=True,
            vis_threshold=0.7,
            neck_ratio_threshold=0.70,
            neck_angle_threshold=70,
            shoulder_height_variation_threshold=0.018,
            put_orientation_text=True,
            resize_image_to=None,
            resize_image_width_to=None,
            resize_image_height_to=None,
            time_bad_posture_alert=5,
            ):
        self.window = window
        self.window.title(window_title)

        self.video_source = video_source
        self.show_video = show_video
        self.auto_detect_orientation = auto_detect_orientation
        self.draw_all_landmarks = draw_all_landmarks
        self.draw_pose_landmarks = draw_pose_landmarks
        self.vis_threshold = vis_threshold
        self.neck_ratio_threshold = neck_ratio_threshold
        self.neck_angle_threshold = neck_angle_threshold
        self.shoulder_height_variation_threshold = shoulder_height_variation_threshold
        self.put_orientation_text = put_orientation_text
        self.resize_image_to = resize_image_to
        self.resize_image_width_to = resize_image_width_to
        self.resize_image_height_to = resize_image_height_to
        self.time_bad_posture_alert = time_bad_posture_alert

        # open video source (by default this will try to open the computer webcam)
        self.cap = MyVideoCapture(self.video_source, show_video)

        if self.resize_image_to is not None:
            self.cap.width = self.resize_image_to[0]
            self.cap.height = self.resize_image_to[1]
        
        if resize_image_width_to is not None or resize_image_height_to is not None:
            width, height = detectPose.image_resize((self.cap.height, self.cap.width, None), width=resize_image_width_to, height=resize_image_height_to)
            self.cap.width, self.cap.height = width, height
        # Create a canvas that can fit the above video source size
        if show_video:
            self.canvas = tk.Canvas(window, width = self.cap.width*2, height=self.cap.height)
        else:
            self.canvas = tk.Canvas(window, width = self.cap.width, height=self.cap.height)
        self.canvas.pack()

        self.btn_width = 10
        self.btn_height = 1
        self.scale_length = 230
        # Button that lets the user take a snapshot
        self.btn_snapshot=tk.Button(window, text="Snapshot", width=self.btn_width, height=self.btn_height, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)
        
        self.btn_toggle_auto_detect_orientation=tk.Button(window, text="Auto Detect", width=self.btn_width, height=self.btn_height, command=self.toggle_auto_detect_orientation)
        self.btn_toggle_auto_detect_orientation.pack(anchor=tk.CENTER, expand=True)

        self.btn_toggle_show_video=tk.Button(window, text="Show Video", width=self.btn_width, height=self.btn_height, command=self.toggle_show_video)
        self.btn_toggle_show_video.pack(anchor=tk.CENTER, expand=True)
        
        self.scale_vis_threshold = tk.Scale(self.window, from_=0, to=99, orient=tk.HORIZONTAL, command=self.change_vis_threshold, length=self.scale_length, label="Visibility Threshold (%)")
        self.scale_vis_threshold.set(int(self.vis_threshold*100))
        self.scale_vis_threshold.pack()
        
        self.scale_neck_ratio_threshold = tk.Scale(self.window, from_=0, to=1, resolution=0.01, digits=3 ,orient=tk.HORIZONTAL, command=self.change_neck_ratio_threshold, length=self.scale_length, label="Neck/Shoulder Ratio Threshold")
        self.scale_neck_ratio_threshold.set(self.neck_ratio_threshold)
        self.scale_neck_ratio_threshold.pack()
        
        self.scale_neck_angle_threshold = tk.Scale(self.window, from_=0, to=90, orient=tk.HORIZONTAL, command=self.change_neck_angle_threshold, length=self.scale_length, label="Neck Angle Threshold (deg)")
        self.scale_neck_angle_threshold.set(self.neck_angle_threshold)
        self.scale_neck_angle_threshold.pack()

        self.scale_shoulder_height_variation_threshold = tk.Scale(self.window, from_=0, to=5, resolution=0.05, digits=3, orient=tk.HORIZONTAL, command=self.change_shoulder_height_variation_threshold, length=self.scale_length, label="Shoulder Height Difference Threshold (%)")
        self.scale_shoulder_height_variation_threshold.set(self.shoulder_height_variation_threshold*100)
        self.scale_shoulder_height_variation_threshold.pack()

        # # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()
        self.window.mainloop()

    def change_vis_threshold(self, value):
        self.vis_threshold = int(value)/100

    def change_neck_ratio_threshold(self, value):
        self.neck_ratio_threshold = float(value)
    
    def change_neck_angle_threshold(self, value):
        self.neck_angle_threshold = int(value)

    def change_shoulder_height_variation_threshold(self, value):
        self.shoulder_height_variation_threshold = float(value)/100

    def toggle_show_video(self):
        self.show_video = not self.show_video
        if self.show_video:
            self.canvas = tk.Canvas(self.window, width=self.cap.width*2, height=self.cap.height)
            self.window.geometry("%dx%d" % (self.cap.width*2, self.window.winfo_height()))
        else:
            self.canvas = tk.Canvas(self.window, width = self.cap.width, height=self.cap.height)
            self.window.geometry("%dx%d" % (self.cap.width, self.window.winfo_height()))
        self.canvas.place(x=0, y=0)

    def toggle_auto_detect_orientation(self):
        self.auto_detect_orientation = not self.auto_detect_orientation

    def snapshot(self):
        # Get a frame from the video source
        ret, frame, frame2 = self.cap.get_frame()
        if not os.path.exists("snapshots"):
            os.makedirs("snapshots")
        if not os.path.exists("snapshots/a"):
            os.mkdir("snapshots/a")
        if not os.path.exists("snapshots/b"):
            os.mkdir("snapshots/b")
        if ret:
            cv2.imwrite("snapshots/a/frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.imwrite("snapshots/b/frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR))

    def update(self):
        # Get a frame from the video source
        ret, frame, frame2 = self.cap.get_frame(
            auto_detect_orientation=self.auto_detect_orientation, 
            show_video=self.show_video,
            draw_all_landmarks=self.draw_all_landmarks,
            draw_pose_landmarks=self.draw_pose_landmarks,
            vis_threshold=self.vis_threshold,
            neck_angle_threshold=self.neck_angle_threshold,
            neck_ratio_threshold=self.neck_ratio_threshold,
            shoulder_height_variation_threshold=self.shoulder_height_variation_threshold,
            put_orientation_text=self.put_orientation_text,
            resize_image_to=self.resize_image_to,
            resize_image_width_to=self.resize_image_width_to,
            resize_image_height_to=self.resize_image_height_to,
            time_bad_posture_alert=self.time_bad_posture_alert,
            )

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.photo2 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame2))

            self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
            self.canvas.create_image(self.cap.width, 0, image = self.photo2, anchor = tk.NW)
        self.window.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self, video_source, show_video):
        # Open the video source
        self.cap = cv2.VideoCapture(video_source)
        self.show_video = show_video

        if not self.cap.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.detector = detectPose(show_video_image=self.show_video)
        self.prev_time = time.time()
        self.time_bad_posture = 0

    def get_frame(self, 
        auto_detect_orientation=False, 
        show_video=False, 
        draw_all_landmarks=True, 
        draw_pose_landmarks=True,
        vis_threshold=0.7,
        neck_angle_threshold=80,
        neck_ratio_threshold=0.70,
        shoulder_height_variation_threshold=0.018,
        put_orientation_text=True,
        resize_image_to=None,
        resize_image_width_to=None,
        resize_image_height_to=None,
        time_bad_posture_alert=5,
        ):
        if self.cap.isOpened():
            ret, image = self.cap.read()
            if not ret:
                print("End video. Ignoring empty camera frame.")
                return (ret, None)
                # Return a boolean success flag and the current frame converted to BGR

            # when false, avoid unnecessary computation
            self.detector.show_video_image = show_video 

            self.detector.set_images(image, 
                resize_image_to=resize_image_to, 
                resize_image_width_to=resize_image_width_to, 
                resize_image_height_to=resize_image_height_to)

            (self.detector.image).flags.writeable = False
            results = self.detector.pose.process(self.detector.image)
            # Draw the pose annotation on the self.image
            (self.detector.image).flags.writeable = True

            self.detector.process_landmarks(results, draw=draw_pose_landmarks, vis_threshold=vis_threshold)

            if draw_all_landmarks:
                self.detector.draw_all_landmarks(results)
            good_posture = self.detector.neck_posture(
                auto_detect_orientation=auto_detect_orientation,
                neck_angle_threshold=neck_angle_threshold,
                neck_ratio_threshold=neck_ratio_threshold,
                shoulder_height_variation_threshold=shoulder_height_variation_threshold,
                put_orientation_text=put_orientation_text)        
            if good_posture:
                self.time_bad_posture = 0

            elif not good_posture:
                self.time_bad_posture += time.time() - self.prev_time
                cv2.putText(self.detector.image, "time bad posture: "+str(round(float(self.time_bad_posture),3)), (0,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
                if self.time_bad_posture > time_bad_posture_alert:
                    cv2.putText(self.detector.image, f"MORE THAN {int(self.time_bad_posture)}s!", (0,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

            self.prev_time = self.detector.show_fps(self.prev_time)

            return (ret, cv2.cvtColor(self.detector.blank_image, cv2.COLOR_BGR2RGB), 
            cv2.cvtColor(self.detector.image, cv2.COLOR_BGR2RGB))

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

# Create a window and pass it to the Application object
if __name__ == "__main__":
    App(tk.Tk(), "Tkinter and OpenCV", 
        video_source=0, 
        # video_source="video_samples/6.mp4",
        show_video=True,
        auto_detect_orientation=False,
        vis_threshold=0.7,
        draw_all_landmarks=False,
        neck_ratio_threshold=0.7,
        neck_angle_threshold=60,
        shoulder_height_variation_threshold=0.018,
        put_orientation_text=True,
        # resize_image_to=(640, 480),
        resize_image_height_to=300,
        # resize_image_width_to=640,
        time_bad_posture_alert=5,
    )