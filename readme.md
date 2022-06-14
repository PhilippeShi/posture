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

# Changing webcam and video inputs
Inside `main.py`, use the following for webcam input:
```python
cap = cv2.VideoCapture(0)
```
or the following for video input:
```python
cap = cv2.VideoCapture("video_samples/1.mp4")
```