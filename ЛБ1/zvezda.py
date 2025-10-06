from math import cos, sin
import numpy as np
from PIL import Image

img_mat=np.zeros((200,200,3),dtype=np.uint8) 

def drawLine(img_mat, x0, y0, x1, y1, count, color):
    step = 1.0/count
    for t in np.arange(0, 1, step):
         x = round((1.0 - t) * x0 + t * x1)
         y = round((1.0 - t) * y0 + t * y1)
         img_mat[y, x] = color

def x_loop_line(img_mat, x0, y0, x1, y1, color, change):

    for x in range (x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        if (change):
            img_mat[x, y] = color
        else:
            img_mat[y, x] = color

def x_loop_line1(img_mat, x0, y0, x1, y1, color, change):
    y = y0
    dy = 2*abs(y1 - y0)
    derror = 0
    y_update = 1 if y1 > y0 else -1
    
    for x in range (x0, x1):
        if (change):
            img_mat[x, y] = color
        else:
            img_mat[y, x] = color
        derror += dy
        if (derror > (x1 - x0)):
            derror -= 2 * (x1 - x0)
            y += y_update
for k in range(13):
    x0, y0 = 100, 100
    x1 = int(100 + cos(2 * 3.14 * k /13) * 95)
    y1 = int(100 + sin(2 * 3.14 * k /13) * 95)
    #drawLine(img_mat, x0, y0, x1, y1, 155, [255, 255, 8])
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    
    x_loop_line1(img_mat, x0, y0, x1, y1, [163, 123, 183], xchange)

img=Image.fromarray(img_mat,mode='RGB')
img.save('img8.png')
