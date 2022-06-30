import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
from pose_module import detectPose

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # open video source (by default this will try to open the computer webcam)
        self.cap = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = self.cap.width, height = self.cap.height)
        self.canvas.pack()

        # Button that lets the user take a snapshot
        self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.cap.get_frame()

        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def update(self):
        # Get a frame from the video source
        ret, frame = self.cap.get_frame()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)

        self.window.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.detector = detectPose(show_video_image=True)
        self.prev_time = time.time()

    def get_frame(self):
        if self.cap.isOpened():
            ret, image = self.cap.read()
            if not ret:
                print("End video. Ignoring empty camera frame.")
                return (ret, None)
                # Return a boolean success flag and the current frame converted to BGR

            self.detector.set_images(image)
            (self.detector.image).flags.writeable = False
            self.detector.image = cv2.cvtColor(self.detector.image, cv2.COLOR_BGR2RGB)
            results = self.detector.pose.process(self.detector.image)
            # Draw the pose annotation on the self.image.
            (self.detector.image).flags.writeable = True
            self.detector.image = cv2.cvtColor((self.detector.image), cv2.COLOR_RGB2BGR)

            self.prev_time = self.detector.show_fps(self.prev_time)
            self.detector.process_landmarks(results, draw=True)
            self.detector.neck_posture(auto_detect=False)        
            self.detector.detect_orientation()
            cv2.imshow("Video", self.detector.image)
            return (ret, cv2.cvtColor(self.detector.blank_image, cv2.COLOR_BGR2RGB))

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

# Create a window and pass it to the Application object
App(tkinter.Tk(), "Tkinter and OpenCV", video_source=0)