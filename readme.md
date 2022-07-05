# Run on Windows
### Install requirements
```powershell
pip install -r requirements.txt
```
### Run main module
```powershell
python main.py
```
<br/>

# Useful parameters in main.py
## Changing webcam and video inputs

```python
# for webcam input
video_source=0

# for video input
video_source="video_samples/1.mp4"
```


## Image resizing
Webcam and video inputs can be resized with the arguments `resize_image_width_to` and/or `resize_image_height_to` in pixels. <br/>
It is preferred to used one <b> one argument</b> to keep the original image ratio.