import sys
from app import App
import tkinter as tk
import json

def main():
    # print(sys.argv)
    # print(type(sys.argv[1]))

    try:
        f=open("settings_main.json","r")
        settings=json.load(f)
        print(settings)
        f.close()
    except:
        settings = {
            "video_source" : 0,
            "show_video" : True,
            "auto_detect_orientation" : True,
            "draw_all_landmarks" : False,
            "draw_pose_landmarks" : True,
            "vis_threshold" : 0.7,
            "neck_ratio_threshold" : 0.7,
            "neck_angle_threshold" : 60,
            "shoulder_height_variation_threshold" : 0.018,
            "put_orientation_text" : True,
            "resize_image_width_to" : None,
            "resize_image_height_to" : 800,
            "time_bad_posture_alert" : 2,
            }

        with open('settings_main.json', 'w') as f:
            json.dump(settings, f)

    App(tk.Tk(), "Tkinter and OpenCV", 
        video_source = settings["video_source"], 
        show_video = settings["show_video"],
        auto_detect_orientation = settings["auto_detect_orientation"],
        vis_threshold = settings["vis_threshold"],
        draw_all_landmarks = settings["draw_all_landmarks"],
        draw_pose_landmarks = settings["draw_pose_landmarks"],
        neck_ratio_threshold = settings["neck_ratio_threshold"],
        neck_angle_threshold = settings["neck_angle_threshold"],
        shoulder_height_variation_threshold = settings["shoulder_height_variation_threshold"],
        put_orientation_text = settings["put_orientation_text"],
        resize_image_height_to = settings["resize_image_height_to"],
        time_bad_posture_alert = settings["time_bad_posture_alert"],
    )

if __name__ == "__main__":
    main()