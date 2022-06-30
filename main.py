import sys
from app import App
import tkinter as tk

def main():
    # print(sys.argv)
    # print(type(sys.argv[1]))

    App(tk.Tk(), "Tkinter and OpenCV", 
    video_source=0, 
    show_video=True, 
    auto_detect_orientation=True,
    vis_threshold=0.7,
    draw_all_landmarks=False,
    neck_ratio_threshold=0.7,
    neck_angle_threshold=60,
    )   
    

if __name__ == "__main__":
    main()