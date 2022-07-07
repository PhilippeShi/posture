import numpy as np

def image_resize(image_dim, width = None, height = None):
    dim = None
    h, w, _ = image_dim

    if width is None and height is None:
        return w,h

    elif (width is not None) and width > w or (height is not None) and height > h:
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