import sys
from app import App
import tkinter as tk

def main():
    # print(sys.argv)
    # print(type(sys.argv[1]))

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
        # resize_image_height_to=480,
        resize_image_width_to=640,
        time_bad_posture_alert=2,
    )
    

if __name__ == "__main__":
    main()