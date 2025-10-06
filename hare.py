import numpy as np
from PIL import Image, ImageOps

img_mat=np.zeros((1000,1000,3),dtype=np.uint8) 

def x_loop_line1(img_mat, x0, y0, x1, y1, color, change):
    y = y0
    dy = 2*abs(y1 - y0)
    derror = 0
    y_update = 1 if y1 > y0 else -1
    
    for x in range (x0, x1):
        if (change):
            img_mat[y, x] = color
        else:
            img_mat[x, y] = color
        derror += dy
        if (derror > (x1 - x0)):
            derror -= 2 * (x1 - x0)
            y += y_update

v = []
f = []
for i in open('e:/обуч/лабораторные/комп граф/ЛБ1п/model_1.obj'):
    if i[0:2] == 'v ':
        v.append(list(map(float, i.split()[1:3])))
    elif i[0:2] == 'f ':
        s = i.split()[1:]
        a = []
        for j in range(3):
            a.append(int(s[j].split('/')[0]))
        f.append(a)

for i in range(len(v)):
    x = int(5000*v[i][0]) + 500
    y = int(5000*v[i][1]) + 250
    img_mat[y, x] = [135, 213, 100]


for k in range(len(f)):
    y0 = int(v[f[k][0]-1][0] * 5000) + 500
    x0 = int(v[f[k][0]-1][1] * 5000) + 250
    y1 = int(v[f[k][1]-1][0] * 5000) + 500
    x1 = int(v[f[k][1]-1][1] * 5000) + 250
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
img = ImageOps.flip(img)
img.save('img_hare.png')
