import numpy as np
from PIL import Image

def load_demo_images():
    ims = []
    for i in range(3):
        im = Image.open('imgs/%d.png' % i)
        ims.append([np.array(im).transpose(
            (2, 0, 1)).astype(np.float32) / 255.])
    return np.array(ims)

load_demo_images()
# print(load_demo_images())
