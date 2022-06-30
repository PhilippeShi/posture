import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk
import time
from pose_module import detectPose

class App:
    def __init__(self, window, window_title, video_source=0, show_video=False, auto_detect=False):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.show_video = show_video
        self.auto_detect = auto_detect
        self.btn_width = 50
        # open video source (by default this will try to open the computer webcam)
        self.cap = MyVideoCapture(self.video_source, show_video, auto_detect)

        self.width = self.cap.width
        self.height = self.cap.height

        # Create a canvas that can fit the above video source size
        if show_video:
            self.canvas = tk.Canvas(window, width = self.width*2, height = self.height)
        else:
            self.canvas = tk.Canvas(window, width = self.width, height = self.height)
        self.canvas.pack()


        # Button that lets the user take a snapshot
        self.btn_snapshot=tk.Button(window, text="Snapshot", width=self.btn_width, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)
        
        self.btn_toggle_auto_detect=tk.Button(window, text="Auto Detect", width=self.btn_width, command=self.toggle_auto_detect)
        self.btn_toggle_auto_detect.pack(anchor=tk.CENTER, expand=True)

        self.btn_toggle_show_video=tk.Button(window, text="Show Video", width=self.btn_width, command=self.toggle_show_video)
        self.btn_toggle_show_video.pack(anchor=tk.CENTER, expand=True)
        
        self.scale = tk.Scale(self.window, from_=0, to=200, orient=tk.HORIZONTAL)
        self.scale.pack()

        self.num_ctrl = 4
        # # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()

    def toggle_show_video(self):
        self.show_video = not self.show_video
        self.cap = MyVideoCapture(self.video_source, self.show_video, self.auto_detect)
        if self.show_video:
            self.canvas = tk.Canvas(self.window, width=self.width*2, height=self.height)
            self.window.geometry("%dx%d" % (self.width*2, self.height + (self.num_ctrl-1)*self.btn_width-10))
        else:
            self.canvas = tk.Canvas(self.window, width = self.width, height = self.height)
            self.window.geometry("%dx%d" % (self.width, self.height + (self.num_ctrl-1)*self.btn_width-10))
        self.canvas.place(x=0, y=0)

    def toggle_auto_detect(self):
        self.auto_detect = not self.auto_detect
        self.cap = MyVideoCapture(self.video_source, self.show_video, self.auto_detect)

    def snapshot(self):
        # Get a frame from the video source
        ret, frame, frame2 = self.cap.get_frame()
        if ret:
            cv2.imwrite("/frames/frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def update(self):
        # Get a frame from the video source
        ret, frame, frame2 = self.cap.get_frame()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.photo2 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame2))

            self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
            self.canvas.create_image(self.cap.width, 0, image = self.photo2, anchor = tk.NW)
        self.window.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self, video_source, show_video, auto_detect):
        # Open the video source
        self.cap = cv2.VideoCapture(video_source)
        self.show_video = show_video
        self.auto_detect = auto_detect

        if not self.cap.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.detector = detectPose(show_video_image=self.show_video)
        self.prev_time = time.time()

    def get_frame(self, auto_detect=False):
        if self.cap.isOpened():
            ret, image = self.cap.read()
            if not ret:
                print("End video. Ignoring empty camera frame.")
                return (ret, None)
                # Return a boolean success flag and the current frame converted to BGR

            self.detector.set_images(image)
            (self.detector.image).flags.writeable = False
            results = self.detector.pose.process(self.detector.image)
            # Draw the pose annotation on the self.image.
            (self.detector.image).flags.writeable = True

            self.prev_time = self.detector.show_fps(self.prev_time)
            self.detector.process_landmarks(results, draw=True)
            self.detector.neck_posture(auto_detect=self.auto_detect)        
            self.detector.detect_orientation()
            return (ret, cv2.cvtColor(self.detector.blank_image, cv2.COLOR_BGR2RGB), 
            cv2.cvtColor(self.detector.image, cv2.COLOR_BGR2RGB))

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

# Create a window and pass it to the Application object
App(tk.Tk(), "Tkinter and OpenCV", video_source=0, show_video=True, auto_detect=False)