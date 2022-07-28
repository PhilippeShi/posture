import numpy as np
import winsound
import os

def image_resize(image_dim, width = None, height = None):
    """
    param:image_dim (height, width, _ )
    return (height, width)
    """
    dim = None
    h, w, _ = image_dim

    if (type(width) is not int) and (type(height) is not int):
        return w,h

    elif (type(width) is int) and width > w or (type(height) is int) and height > h:
        return w,h

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return dim

def get_angle(a:list, b:list, c:list, dimensions:int, decimals:int=2, less_than_180:bool=True):

    a = np.array(a[:dimensions])
    b = np.array(b[:dimensions])
    c = np.array(c[:dimensions])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    rad_angle = np.arccos(cosine_angle)
    deg_angle = np.degrees(rad_angle)

    if less_than_180 and deg_angle > 180:
        deg_angle = 360 - deg_angle

    return round(deg_angle,decimals)

def get_distance(a:list, b:list, dimensions:int=2):
    a = np.array(a[:dimensions])
    b = np.array(b[:dimensions])
    return np.linalg.norm(a - b)

class sound_alert:
    def __init__(self):
        self.prev = "good"
        dir = os.path.dirname(os.path.realpath(__file__))
        self.path = os.path.join(dir, "beep.wav")

    def sound_alert(self, msg):
        if "ALERT" in msg and self.prev == "good":
            winsound.PlaySound(self.path, winsound.SND_LOOP | winsound.SND_ASYNC)
            self.prev = "bad"

        elif "good" in msg:
            self.prev = "good"
            winsound.PlaySound(None, winsound.SND_PURGE)

if __name__ == "__main__":
    d = sound_alert()
    print(d.path)
    d.sound_alert("ALERT")
