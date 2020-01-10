import cv2
import numpy as np
from glob import iglob

TIME = 1  # Time in secs to display each frame

imgs = []
for img in iglob("output/*"):
    imgs.append(cv2.imread(img))

out = cv2.VideoWriter('output.avi',
                      cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                      30, (640, 480))
for img in imgs:
    for i in range(TIME * 30):
        out.write(img)
