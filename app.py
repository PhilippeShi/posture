import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk
import time
import numpy as np

from detect_posture.pose import detectPose
from detect_posture.utils import image_resize
from detect_posture.utils import sound_alert
from network import client
import json
import mediapipe as mp

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
            shoulder_hip_ratio_threshold=0.4,
            put_orientation_text=True,
            resize_image_width_to=None,
            resize_image_height_to=None,
            time_bad_posture_alert=5,
            show_fps=True,
            mirror_mode=True,
            alert_sound=True,
            alert_other_device=False,
            ip_address=None,
            ):

        self.window = window
        self.window.title(window_title)
        
        self.show_video = show_video
        self.auto_detect_orientation = auto_detect_orientation
        self.draw_all_landmarks = draw_all_landmarks
        self.draw_pose_landmarks = draw_pose_landmarks
        self.vis_threshold = vis_threshold
        self.neck_ratio_threshold = neck_ratio_threshold
        self.neck_angle_threshold = neck_angle_threshold
        self.shoulder_height_variation_threshold = shoulder_height_variation_threshold
        self.shoulder_hip_ratio_threshold = shoulder_hip_ratio_threshold
        self.put_orientation_text = put_orientation_text
        self.resize_image_width_to = resize_image_width_to
        self.resize_image_height_to = resize_image_height_to
        self.time_bad_posture_alert = time_bad_posture_alert
        self.alert_sound = alert_sound
        self.show_fps = show_fps
        self.mirror_mode = mirror_mode
        self.add_bad_posture_flag = False

        self.kwargs_frame = {
            "show_video" : self.show_video,
            "auto_detect_orientation" : self.auto_detect_orientation, 
            "draw_all_landmarks" : self.draw_all_landmarks,
            "draw_pose_landmarks" : self.draw_pose_landmarks,
            "vis_threshold" : self.vis_threshold,
            "neck_ratio_threshold" : self.neck_ratio_threshold,
            "neck_angle_threshold" : self.neck_angle_threshold,
            "shoulder_height_variation_threshold" : self.shoulder_height_variation_threshold,
            "shoulder_hip_ratio_threshold" : self.shoulder_hip_ratio_threshold,
            "put_orientation_text" : self.put_orientation_text,
            "resize_image_width_to" : self.resize_image_width_to,
            "resize_image_height_to" : self.resize_image_height_to,
            "time_bad_posture_alert" : self.time_bad_posture_alert,
            "alert_sound" : self.alert_sound,
            "show_fps" : self.show_fps,
            "mirror_mode" : self.mirror_mode,
            "add_bad_posture_flag" : False,
        }
        self.kwargs_other = {
            "video_source" : video_source,
            "alert_other_device" : alert_other_device,
            "ip_address" : ip_address,
        }

        self.cap = MyVideoCapture(video_source, show_video, alert_other_device, ip_address)

        self.all_widgets = {}
        self.neck_widgets = []
        self.shown_widgets = []

        width, height = image_resize((self.cap.height, self.cap.width, None), width=resize_image_width_to, height=resize_image_height_to)
        self.cap.width, self.cap.height = width, height

        # Create a canvas that can fit the above video source size
        if show_video:
            self.canvas = tk.Canvas(window, width = self.cap.width*2, height=self.cap.height)
        else:
            self.canvas = tk.Canvas(window, width = self.cap.width, height=self.cap.height)
        
        self.all_widgets["Canvas"] = self.canvas
        self.shown_widgets.append(self.canvas)

        btn_width = 20
        scale_length = 230

        self.slide_crop_image=tk.Scale(self.window, from_=0, to=self.cap.width, orient=tk.HORIZONTAL, length=scale_length, command=self.crop_image_location, label="Crop Image X-Axis")
        self.all_widgets["Crop Image X-Axis"] = self.slide_crop_image
        self.shown_widgets.append(self.slide_crop_image)

        self.slide_crop_image_width=tk.Scale(self.window, from_=150, to=self.cap.width, orient=tk.HORIZONTAL, length=scale_length, command=self.crop_image_width, label="Crop Image Width")
        self.all_widgets["Crop Image Width"] = self.slide_crop_image_width
        self.shown_widgets.append(self.slide_crop_image_width)

        self.btn_toggle_crop_image = tk.Button(self.window, text="Toggle Crop Image", width=btn_width, command=self.toggle_crop_image)
        self.all_widgets["Toggle Crop Image"] = self.btn_toggle_crop_image
        self.shown_widgets.append(self.btn_toggle_crop_image)
      
        self.btn_toggle_auto_detect_orientation=tk.Button(self.window, text="Auto Detect", width=btn_width, command=self.toggle_auto_detect_orientation)
        self.all_widgets["Auto Detect"] = self.btn_toggle_auto_detect_orientation
        self.shown_widgets.append(self.btn_toggle_auto_detect_orientation)

        self.btn_toggle_show_video=tk.Button(self.window, text="Show Video", width=btn_width, command=self.toggle_show_video)
        self.all_widgets["Show Video"] = self.btn_toggle_show_video
        self.shown_widgets.append(self.btn_toggle_show_video)

        self.btn_save_settings=tk.Button(self.window, text="Save Settings", width=btn_width, command=self.save_settings)
        self.all_widgets["Save Settings"] = self.btn_save_settings
        self.shown_widgets.append(self.btn_save_settings)
        
        self.btn_toggle_neck_widgets=tk.Button(self.window, text="Neck Settings", width=btn_width, command=self.neck_settings)
        self.all_widgets["Neck Settings"] = self.btn_toggle_neck_widgets
        self.shown_widgets.append(self.btn_toggle_neck_widgets)

        self.scale_vis_threshold = tk.Scale(self.window, from_=0, to=99, orient=tk.HORIZONTAL, command=self.change_vis_threshold, length=scale_length, label="Visibility Threshold (%)")
        self.scale_vis_threshold.set(int(self.vis_threshold*100))
        self.all_widgets["Visibility Threshold"] = self.scale_vis_threshold
        self.shown_widgets.append(self.scale_vis_threshold)

        self.btn_add_bad_posture = tk.Button(self.window, text="Add Bad Posture", width=btn_width, command=self.add_bad_posture)
        self.all_widgets["Add Bad Posture"] = self.btn_add_bad_posture
        self.shown_widgets.append(self.btn_add_bad_posture)
        
        self.scale_neck_ratio_threshold = tk.Scale(self.window, from_=0, to=1, resolution=0.01, digits=3 ,orient=tk.HORIZONTAL, command=self.change_neck_ratio_threshold, length=scale_length, label="Neck/Shoulder Ratio Threshold")
        self.scale_neck_ratio_threshold.set(self.neck_ratio_threshold)
        self.neck_widgets.append(self.scale_neck_ratio_threshold)
        self.all_widgets["Neck/Shoulder Threshold"] = self.scale_neck_ratio_threshold
        
        self.scale_neck_angle_threshold = tk.Scale(self.window, from_=0, to=90, orient=tk.HORIZONTAL, command=self.change_neck_angle_threshold, length=scale_length, label="Neck Angle Threshold (deg)")
        self.scale_neck_angle_threshold.set(self.neck_angle_threshold)
        self.neck_widgets.append(self.scale_neck_angle_threshold)
        self.all_widgets["Neck Angle Threshold (deg)"] = self.scale_neck_angle_threshold

        self.scale_shoulder_hip_ratio_threshold = tk.Scale(self.window, from_=0, to=1, resolution=0.01, digits=3, orient=tk.HORIZONTAL, command=self.change_shoulder_hip_ratio_threshold, length=scale_length, label="Shoulder Hip Ratio Threshold")
        self.scale_shoulder_hip_ratio_threshold.set(self.shoulder_hip_ratio_threshold)
        self.neck_widgets.append(self.scale_shoulder_hip_ratio_threshold)
        self.all_widgets["Shoulder Hip Ratio Threshold"] = self.change_shoulder_hip_ratio_threshold
        
        # self.scale_shoulder_height_variation_threshold = tk.Scale(self.window, from_=0, to=5, resolution=0.05, digits=3, orient=tk.HORIZONTAL, command=self.change_shoulder_height_variation_threshold, length=scale_length, label="Shoulder Height Difference Threshold (%)")
        # self.scale_shoulder_height_variation_threshold.set(self.shoulder_height_variation_threshold*100)
        # self.neck_widgets.append(self.scale_shoulder_height_variation_threshold)
        # self.all_widgets["Shoulder Height Difference Threshold (%)"] = self.scale_shoulder_height_variation_threshold

        clicked = tk.StringVar()
        
        self.menu = tk.OptionMenu(self.window, clicked, *self.all_widgets.keys(), command=self.option_changed)
        self.shown_widgets.append(self.menu)

        self.initialize(self.shown_widgets)
        self.neck_widgets_shown = False

        # # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()
        self.window.mainloop()

    def initialize(self, widgets):
        for widget in widgets:
            widget.pack()
    
    def forget_all(self, widgets):
        for widget in widgets:
            widget.pack_forget()

    def option_changed(self, option):
        # use match-case instead of if-statements for python 3.10 and above
        # I am currently using python 3.9.5
        if option == "Neck Settings":
            self.neck_settings()
        if option == "Show Video":
            self.toggle_show_video()
        if option == "Auto Detect":
            self.toggle_auto_detect_orientation()
        if option == "Save Settings":
            self.save_settings()
        

    def save_settings(self):
        d = self.kwargs_frame.copy()
        d.update(self.kwargs_other)
        with open('settings.json', 'w') as f:
            json.dump({**self.kwargs_other, **self.kwargs_frame}, f)
        print("Settings saved")

    def neck_settings(self):
        if self.neck_widgets_shown is False:
            self.neck_widgets_shown = True
            for w in self.neck_widgets:
                w.pack()
        else:
            self.neck_widgets_shown = False
            for w in self.neck_widgets:
                w.forget()
    
    def crop_image_location(self, value):
        self.kwargs_frame["crop"] = int(value)
        self.crop_x = int(value)
    
    def crop_image_width(self, value):
        self.kwargs_frame["crop_width"] = int(value)

    def toggle_crop_image(self):
        if self.kwargs_frame["crop"] is None:
            self.kwargs_frame["crop"] = self.crop_x
        else:
            self.kwargs_frame["crop"] = None

    def add_bad_posture(self):
        self.kwargs_frame["add_bad_posture_flag"] = True

    def change_vis_threshold(self, value):
        self.kwargs_frame["vis_threshold"] = int(value)/100

    def change_neck_ratio_threshold(self, value):
        self.kwargs_frame["neck_ratio_threshold"] = float(value)
    
    def change_neck_angle_threshold(self, value):
        self.kwargs_frame["neck_angle_threshold"] = int(value)

    def change_shoulder_height_variation_threshold(self, value):
        self.kwargs_frame["shoulder_height_variation_threshold"] = float(value)/100
    
    def change_shoulder_hip_ratio_threshold(self, value):
        self.kwargs_frame["shoulder_hip_ratio_threshold"] = float(value)

    def toggle_show_video(self):
        self.kwargs_frame["show_video"] = not self.kwargs_frame["show_video"]
        if self.kwargs_frame["show_video"]:
            self.canvas = tk.Canvas(self.window, width=self.cap.width*2, height=self.cap.height)
            self.window.geometry("%dx%d" % (self.cap.width*2, self.window.winfo_height()))
        else:
            self.canvas = tk.Canvas(self.window, width = self.cap.width, height=self.cap.height)
            self.window.geometry("%dx%d" % (self.cap.width, self.window.winfo_height()))
        self.canvas.place(x=0, y=0)

    def toggle_auto_detect_orientation(self):
        self.kwargs_frame["auto_detect_orientation"] = not self.kwargs_frame["auto_detect_orientation"]
        # self.auto_detect_orientation = not self.auto_detect_orientation

    def update(self):
        # Get a frame from the video source
        ret, frame, frame2 = self.cap.get_frame(**self.kwargs_frame)
        self.kwargs_frame["add_bad_posture_flag"] = False
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.photo2 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame2))

            self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
            self.canvas.create_image(self.cap.width, 0, image = self.photo2, anchor = tk.NW)
        self.window.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self, video_source, show_video, alert_other_device=False, ip=None):
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
        self.alert_other_device = alert_other_device
        self.sound_alert = sound_alert()
        self.detector2 = detectPose(show_video_image=self.show_video)
        
        if alert_other_device:
            if ip is None:
                self.client = client.Client()
            else:
                self.client = client.Client(ip)

    def get_frame(self,
        show_video=False,
        auto_detect_orientation=False,
        draw_all_landmarks=True,
        draw_pose_landmarks=True,
        vis_threshold=0.7,
        neck_ratio_threshold=0.70,
        neck_angle_threshold=80,
        shoulder_height_variation_threshold=0.018,
        shoulder_hip_ratio_threshold=0.4,
        put_orientation_text=True,
        resize_image_width_to=None,
        resize_image_height_to=None,
        time_bad_posture_alert=5,
        alert_sound=True,
        show_fps=True,
        mirror_mode=True,
        add_bad_posture_flag=False,
        crop=None,
        crop_width=150,
        ):
        if self.cap.isOpened():
            ret, image = self.cap.read()
            if not ret:
                print("End video. Ignoring empty camera frame.")
                return (ret, None)
                # Return a boolean success flag and the current frame converted to BGR

            # when false, avoid unnecessary computation
            self.detector.show_video_image = show_video 
            

            if mirror_mode:
                image = cv2.flip(image, 1) 
            
            if crop is not None:
                og_image = image.copy()
                og_image = cv2.blur(og_image, (50,50))
                og_blank_image = np.zeros(image.shape, dtype=np.uint8)
                image = image[0:self.height, crop:crop+crop_width]

            self.detector.set_images(image, 
            resize_image_width_to=resize_image_width_to, 
            resize_image_height_to=resize_image_height_to)
            

            results = self.detector.pose.process(self.detector.image)
            self.detector.process_landmarks(results, draw=draw_pose_landmarks, vis_threshold=vis_threshold)
            # self.detector.blur_person()
            if draw_all_landmarks:
                self.detector.draw_all_landmarks(results)

            # self.detector2.set_images(self.detector.image)
            # results2 = self.detector2.pose.process(self.detector2.image)
            # self.detector2.process_landmarks(results2, draw=draw_all_landmarks, vis_threshold=vis_threshold)
            # self.detector2.blur_person()



            # if draw_all_landmarks:
            #     self.detector2.draw_all_landmarks(results2)

            # cv2.imshow("image", self.detector2.image)

            good_posture, ratio, angle = True, 0, 0
            try:
                good_posture, _, ratio, angle, _ = self.detector.neck_posture(
                    auto_detect_orientation=auto_detect_orientation,
                    neck_angle_threshold=neck_angle_threshold,
                    neck_ratio_threshold=neck_ratio_threshold,
                    shoulder_height_variation_threshold=shoulder_height_variation_threshold,
                    shoulder_hip_ratio_threshold=shoulder_hip_ratio_threshold,
                    put_orientation_text=put_orientation_text,)     
            except:
                pass
            
            if good_posture is True:
                good_posture = not self.detector.check_bad_posture(ratio, angle)
                if good_posture is False:
                    print("Bad posture detected from added posture")
            # else:
            #     print("bad posture")
            # if good_posture:
            #     print("GOOD")

            if add_bad_posture_flag:
                self.detector.set_bad_posture(ratio, angle)

            send_msg = ""
            
            # every half-second
            # if (time.time()%60%1) <= 0.6 and (time.time()%60%1) >= 0.5 and self.prev_time%60%1 <0.5:
            #     print(self.prev_time%60%1, time.time()%60%1)

            if good_posture:
                self.time_bad_posture = 0
                send_msg = "good posture"

            elif not good_posture:
                self.time_bad_posture += time.time() - self.prev_time
                for img in self.detector.images():
                    cv2.putText(img, "time: "+str(round(float(self.time_bad_posture),2)), (0,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                if self.time_bad_posture > time_bad_posture_alert:
                    for img in self.detector.images():
                        cv2.putText(img, f"ALERT {int(self.time_bad_posture)}s!", (0,150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    
                    send_msg = "ALERT"
                    
                else:
                    send_msg = "bad posture"

            # at every second or half-second:
            if abs(int(time.time()%60) - int(self.prev_time%60)) >= 1 or ((time.time()%60%1) <= 0.6 and (time.time()%60%1) >= 0.5 and self.prev_time%60%1 <0.5): 
                if self.alert_other_device:
                    if alert_sound:
                        self.client.send(f"{send_msg} {int(self.time_bad_posture)} sound")
                    else:
                        self.client.send(f"{send_msg} {int(self.time_bad_posture)}")

                elif alert_sound:
                    self.sound_alert.sound_alert(send_msg)

            if show_fps:
                self.prev_time = self.detector.show_fps(self.prev_time)
            else:
                self.prev_time = time.time()

            if crop is not None:

                og_image[0:self.detector.image.shape[0], crop:crop+self.detector.image.shape[1]] = self.detector.image
                og_blank_image[0:self.detector.blank_image.shape[0], crop:crop+self.detector.blank_image.shape[1]] = self.detector.blank_image
                return (ret, cv2.cvtColor(og_blank_image, cv2.COLOR_BGR2RGB), 
                        cv2.cvtColor(og_image, cv2.COLOR_BGR2RGB))

            return (ret, cv2.cvtColor(self.detector.blank_image, cv2.COLOR_BGR2RGB), 
            cv2.cvtColor(self.detector.image, cv2.COLOR_BGR2RGB))

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.alert_other_device:
            self.client.close()
        if self.cap.isOpened():
            self.cap.release()

# Create a window and pass it to the Application object
if __name__ == "__main__":
    settings = {
        "video_source" : 0,
        # "video_source" : "video_samples/4.mp4",
        "show_video" : True,
        "auto_detect_orientation" : True,
        "draw_all_landmarks" : True,
        "draw_pose_landmarks" : True,
        "vis_threshold" : 0.7,
        "neck_ratio_threshold" : 0.65,
        "neck_angle_threshold" : 60,
        "shoulder_height_variation_threshold" : 0.018,
        "shoulder_hip_ratio_threshold" : 0.45,
        "put_orientation_text" : True,
        "resize_image_width_to" : 600,
        "resize_image_height_to" : None,
        "time_bad_posture_alert" : 2,
        "show_fps" : False,
        "mirror_mode" : True,
        "alert_other_device": False,
        "alert_sound": False,
        "ip_address": None,
        }

    App(tk.Tk(), "Tkinter and OpenCV", **settings)

    