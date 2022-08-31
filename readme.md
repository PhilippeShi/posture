
# Run on Windows
To use the program, simply run the [main.exe](dist/main.exe) file. <br/>
If you prefer using a python environment, install the requirements and run the main file
```powershell
pip install -r requirements.txt
python main.py
```
<br/>

# Settings 
When first running the main app, a `settings.json` file will automatically be created in the directory where it is being run with the default settings. <br/>
| Setting 	| Description 	| Value Type 	| Default value and examples	| Nullable
|---------	|-------------	|------------	|----------	|----------
|video_source|You can choose which webcam to use by its index. If your computer only has one webcam, then its value is probably 0. You can also choose to use a video as input.|Int <br/> "String"|0<br/>"path/video.mp4"| No
|show_fps|Show how many frames per second are being analyzed. |Boolean|false|No
|show_video|Show posture detections on the original video source (true) or on a blank canvas (false)|Boolean|true|No
|mirror_mode|Inverts the camera video input horizontally.|Boolean|true|No
|auto_detect_orientation|Automatically detects if the user is facing fowards or sideways. Needs to be true to have automatic alerts for bad postures|Boolean|true|No
|draw_all_landmarks|Show all body landmarks that can be detected.|Boolean|false|No
|draw_pose_landmarks|Show only necessary body landmarks to visualize body posture.|Boolean|true|No
|vis_threshold|Minimum confidence level of a landmark to be considered as being detected, between 0 and 1.|Float|0.7|No
|put_orientation_text|Show whether the user is detected to be facing fowards or sideways. The sensitivity can be adjusted with the next two settings.|Boolean|true|No
|shoulder_hip_ratio_threshold|Default method of detecting orientation. Uses the ratio of shoulder-to-shoulder ditance over hip-to-hip distance. Detects facing fowards if value is over the threshold, and facing sideways if under.|Float|0.45|Yes
|shoulder_height_variation_threshold|Less accurate method of detecting orientation. Uses the height difference between the two shoulders. Detects facing fowards if value is under the threshold, and facing sideways if over. Used ONLY when the previous value shoulder_hip_ratio_threshold is null.|Float|0.018|No
|neck_ratio_threshold|Only when facing fowards. <br/>Uses the ratio of neck length over shoulder-to-shoulder distance. Detects bad posture when ratio is under the threshold.|Float|0.65|No 
|neck_angle_threshold|Only when facing sideways. <br/>Uses the smallest angle made from shoulder-to-neck and neck-to-nose angle. Detects bad posture when angle is under the threshold.|Int|60|No
|resize_image_width_to|Resize image width in pixels. Preferably to be used without resize_image_height_to to keep original aspect ratio.|Int|400|Yes
|resize_image_height_to|Resize image height in pixels. Preferably to be used without resize_image_width_to to keep original aspect ratio.|Int|null|Yes
|time_bad_posture_alert|Alert the user only if the user is in bad posture for over certain amount of seconds.|Int|3|No
|alert_sound|When alerting the user, also play a sound.|Boolean|true|No
|alert_other_device|Alert the user on another device. The other device has to run this [program](dist/server.exe). More info [here](#receiving-alerts-on-another-device).|Boolean|true|No
|ip_address|Only when alert_other_device is true. Has to be the same IPv4 address which the second device server is listening on. More info [here](#receiving-alerts-on-another-device).|String|null <br/> "10.111.200.200"|Yes

<br/>

# Receiving alerts on another device
On your second device, download the following executable file [server.exe](dist/server.exe), or if you prefer as a python file [server.py](network/server.py). When running it, the IPv4 address in which the server listens to will be shown on the console. Use that address in the ip_address settings.

<br/>

# Creating executable file
## Option 1
In the root project folder, run the following command:
```powershell
pyinstaller main.spec
```
## Option 2
```powershell
pyinstaller -F main.py
``` 
then inside the `main.spec` file, add the following snipped under `"block_cipher = None"`:
```
def get_mediapipe_path():
    import mediapipe
    mediapipe_path = mediapipe.__path__[0]
    return mediapipe_path
```
and add the following under `"pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)"`
```
mediapipe_tree = Tree(get_mediapipe_path(), prefix='mediapipe', excludes=["*.pyc"])
a.datas += mediapipe_tree
a.binaries = filter(lambda x: 'mediapipe' not in x[0], a.binaries)
```

Now, run to create the executable file
```powershell
pyinstaller main.spec
```
<br/>

## Random info
[How large files are tracked using Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/configuring-git-large-file-storage)
