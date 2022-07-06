
# Run on Windows
### Install requirements and run main file
```powershell
pip install -r requirements.txt
python main.py
```
<br/>

## Useful parameters in main.py
### Changing webcam and video inputs

```python
# for webcam input
video_source=0

# for video input
video_source="video_samples/1.mp4"
```


### Image resizing
Webcam and video inputs can be resized with the arguments `resize_image_width_to` and/or `resize_image_height_to` in pixels. <br/>
It is preferred to used one <b> one argument</b> to keep the original image ratio.
<br/>
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

Install the executable file [here](dist/main.exe)

[How large files are tracked using Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/configuring-git-large-file-storage)
